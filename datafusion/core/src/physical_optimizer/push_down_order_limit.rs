// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! CombinePartialFinalAggregate optimizer rule checks the adjacent Partial and Final AggregateExecs
//! and try to combine them if necessary

use itertools::Itertools;
use std::sync::Arc;

use crate::error::Result;
use crate::physical_optimizer::PhysicalOptimizerRule;
use datafusion_common::config::ConfigOptions;
use datafusion_common::tree_node::{Transformed, TreeNode};
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::LexOrdering;
use datafusion_physical_plan::aggregates::{AggregateExec, AggregateMode};
use datafusion_physical_plan::coalesce_batches::CoalesceBatchesExec;
use datafusion_physical_plan::filter::FilterExec;
use datafusion_physical_plan::repartition::RepartitionExec;
use datafusion_physical_plan::sorts::sort::SortExec;
use datafusion_physical_plan::ExecutionPlan;

#[derive(Default)]
pub struct PushDownOderLimit {}

impl PushDownOderLimit {
    #[allow(missing_docs)]
    pub fn new() -> Self {
        Self {}
    }

    fn transform_partial_agg(
        aggr: &AggregateExec,
        sort: &SortExec,
    ) -> Option<Arc<dyn ExecutionPlan>> {
        // let (field, desc) = aggr.get_minmax_desc()?;

        // let group_key = aggr.group_expr().expr().iter().exactly_one().ok()?;
        // let kt = group_key.0.data_type(&aggr.input().schema()).ok()?;
        // if !kt.is_primitive() && kt != DataType::Utf8 {
        //     return None;
        // }

        // if aggr.filter_expr().iter().any(|e| e.is_some()) {
        //     return None;
        // }

        // // ensure the sort is on the same field as the aggregate output
        // let col = order.expr.as_any().downcast_ref::<Column>()?;
        // if col.name() != field.name() {
        //     return None;
        // }

        // We found what we want: clone, copy the limit down, and return modified node
        let new_aggr = AggregateExec::try_new(
            *aggr.mode(),
            aggr.group_by().clone(),
            aggr.aggr_expr().to_vec(),
            aggr.filter_expr().to_vec(),
            aggr.input().clone(),
            aggr.input_schema(),
        )
        .expect("Unable to copy Aggregate!")
        .with_limit(sort.fetch())
        .with_ordering(reorder(sort, aggr));
        Some(Arc::new(new_aggr))
    }

    fn transform_sort_limit(
        plan: Arc<dyn ExecutionPlan>,
    ) -> Option<Arc<dyn ExecutionPlan>> {
        let sort = plan.as_any().downcast_ref::<SortExec>()?;

        let children = sort.children();
        let child = children.iter().exactly_one().ok()?;

        let is_cardinality_preserving = |plan: Arc<dyn ExecutionPlan>| {
            plan.as_any()
                .downcast_ref::<CoalesceBatchesExec>()
                .is_some()
                || plan.as_any().downcast_ref::<RepartitionExec>().is_some()
                || plan.as_any().downcast_ref::<FilterExec>().is_some()
        };

        let mut cardinality_preserved = true;
        let mut closure = |plan: Arc<dyn ExecutionPlan>| {
            if !cardinality_preserved {
                return Ok(Transformed::No(plan));
            }
            if let Some(aggr) = plan.as_any().downcast_ref::<AggregateExec>() {
                if matches!(aggr.mode(), AggregateMode::Partial)
                    && can_push_down(aggr, sort)
                {
                    // either we run into an Aggregate and transform it
                    match Self::transform_partial_agg(aggr, sort) {
                        None => cardinality_preserved = false,
                        Some(plan) => return Ok(Transformed::Yes(plan)),
                    }
                } else {
                    return Ok(Transformed::No(plan));
                }
            } else {
                // or we continue down whitelisted nodes of other types
                if !is_cardinality_preserving(plan.clone()) {
                    cardinality_preserved = false;
                }
            }
            Ok(Transformed::No(plan))
        };
        let child = child.clone().transform_down_mut(&mut closure).ok()?;
        let sort = SortExec::new(sort.expr().to_vec(), child)
            .with_fetch(sort.fetch())
            .with_preserve_partitioning(sort.preserve_partitioning());
        Some(Arc::new(sort))
    }
}

impl PhysicalOptimizerRule for PushDownOderLimit {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let plan = plan.transform_down(&|plan| {
            Ok(
                if let Some(plan) = PushDownOderLimit::transform_sort_limit(plan.clone())
                {
                    Transformed::Yes(plan)
                } else {
                    Transformed::No(plan)
                },
            )
        })?;
        Ok(plan)
    }

    fn name(&self) -> &str {
        "PushDownOderLimit"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

fn reorder(sort: &SortExec, aggr: &AggregateExec) -> Option<LexOrdering> {
    let old_group_expr = aggr
        .group_expr()
        .expr()
        .iter()
        .map(|(_, name)| name.as_str())
        .collect::<Vec<_>>();

    let mut common_columns = Vec::new();

    sort.expr()
        .iter()
        .map(|expr| (expr.expr.as_any().downcast_ref::<Column>().unwrap(), expr))
        .for_each(|(column, expr)| {
            if old_group_expr.contains(&column.name()) {
                common_columns.push(expr.clone());
            }
        });
    Some(common_columns)
}

fn can_push_down(aggr: &AggregateExec, sort: &SortExec) -> bool {
    let columns = aggr
        .group_expr()
        .expr()
        .iter()
        .map(|(_, name)| name.as_str())
        .collect::<Vec<_>>();

    sort.expr()
        .iter()
        .filter(|expr| {
            if let Some(column) = expr.expr.as_any().downcast_ref::<Column>() {
                columns.contains(&column.name())
            } else {
                false
            }
        })
        .count()
        > 0
}
