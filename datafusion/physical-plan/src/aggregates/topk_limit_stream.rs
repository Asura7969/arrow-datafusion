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

//! A memory-conscious aggregation implementation that limits group buckets to a fixed number

use crate::aggregates::{
    aggregate_expressions, evaluate_group_by, evaluate_many, AggregateExec,
    PhysicalGroupBy,
};
use crate::sorts::sort::sort_batch;
use crate::{ExecutionPlan, RecordBatchStream, SendableRecordBatchStream};
use arrow::compute::concat_batches;
use arrow::util::pretty::print_batches;
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::SchemaRef;
use datafusion_common::DataFusionError;
use datafusion_common::Result;
use datafusion_execution::TaskContext;
use datafusion_physical_expr::{PhysicalExpr, PhysicalSortExpr};
use futures::stream::{Stream, StreamExt};
use log::{trace, Level};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

pub struct GroupedTopKLimitAggregateStream {
    partition: usize,
    row_count: usize,
    schema: SchemaRef,
    input: SendableRecordBatchStream,
    aggregate_arguments: Vec<Vec<Arc<dyn PhysicalExpr>>>,
    group_by: PhysicalGroupBy,
    limit: Option<usize>,
    in_mem_batches: Vec<RecordBatch>,
    /// Sort expressions
    order: Vec<PhysicalSortExpr>,
}

impl GroupedTopKLimitAggregateStream {
    pub fn new(
        aggr: &AggregateExec,
        context: Arc<TaskContext>,
        partition: usize,
    ) -> Result<Self> {
        let agg_schema = Arc::clone(&aggr.schema);
        let group_by = aggr.group_by.clone();
        let input = aggr.input.execute(partition, Arc::clone(&context))?;
        let aggregate_arguments =
            aggregate_expressions(&aggr.aggr_expr, &aggr.mode, group_by.expr.len())?;

        let order = aggr
            .output_ordering()
            .ok_or_else(|| DataFusionError::Internal("ordering required".to_string()))?
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        let order = order.clone();
        let limit = aggr.limit();

        Ok(GroupedTopKLimitAggregateStream {
            partition,
            row_count: 0,
            schema: agg_schema,
            input,
            aggregate_arguments,
            group_by,
            limit,
            in_mem_batches: Vec::new(),
            order,
        })
    }
}

impl RecordBatchStream for GroupedTopKLimitAggregateStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

impl GroupedTopKLimitAggregateStream {
    fn sort_in_mem_batches(self: &mut Pin<&mut Self>) -> Result<()> {
        let input_batch = concat_batches(&self.schema(), &self.in_mem_batches)?;
        self.in_mem_batches.clear();
        let batch = sort_batch(&input_batch, &self.order, self.limit)?;
        self.row_count += batch.num_rows();
        if log::log_enabled!(Level::Trace) && batch.num_rows() < 20 {
            print_batches(&[batch.clone()])?;
        }
        self.in_mem_batches.push(batch);
        Ok(())
    }
}

impl Stream for GroupedTopKLimitAggregateStream {
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        while let Poll::Ready(res) = self.input.poll_next_unpin(cx) {
            match res {
                // got a batch, convert to rows and append to our TreeMap
                Some(Ok(batch)) => {
                    // println!("{:?}", batch.schema());
                    // println!("{:?}", self.schema());
                    trace!(
                        "partition {} has {} rows and got batch with {} rows",
                        self.partition,
                        self.row_count,
                        batch.num_rows()
                    );
                    let batches = &[batch];

                    // println!("group by: {:?}", &self.group_by);
                    // println!("aggregate arguments: {:?}", &self.aggregate_arguments);

                    let group_by_values =
                        evaluate_group_by(&self.group_by, batches.first().unwrap())?;
                    let input_values = evaluate_many(
                        &self.aggregate_arguments,
                        batches.first().unwrap(),
                    )?;
                    // TODO: 1、性能问题 2、按order by 条件过滤比已有数据小的列，减少聚合数据
                    let mut cols = group_by_values.into_iter().flatten().collect::<Vec<ArrayRef>>();
                    let mut values = input_values.into_iter().flatten().collect::<Vec<ArrayRef>>();
                    cols.append(&mut values);

                    let batch = RecordBatch::try_new(self.schema.clone(), cols)?;
                    self.in_mem_batches.push(batch);

                    self.sort_in_mem_batches()?
                }
                // inner is done, emit all rows and switch to producing output
                None => {
                    return if let Some(batch) = self.in_mem_batches.pop() {
                        trace!(
                            "partition {} emit batch with {} rows",
                            self.partition,
                            batch.num_rows()
                        );
                        if log::log_enabled!(Level::Trace) {
                            print_batches(&[batch.clone()])?;
                        }
                        Poll::Ready(Some(Ok(batch.clone())))
                    } else {
                        trace!("partition {} emit None", self.partition);
                        Poll::Ready(None)
                    }
                }
                // inner had error, return to caller
                Some(Err(e)) => {
                    return Poll::Ready(Some(Err(e)));
                }
            }
        }
        Poll::Pending
    }
}
