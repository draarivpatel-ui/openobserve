// Copyright 2026 OpenObserve Inc.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

use std::{
    iter::zip,
    sync::{Arc, LazyLock as Lazy},
};

use config::tantivy::tokenizer::o2_collect_search_tokens;
use datafusion::{
    arrow::{
        array::{ArrayRef, BooleanArray, Int64Array, StringArrayType},
        datatypes::DataType,
    },
    common::{cast::as_int64_array, utils::datafusion_strsim::levenshtein},
    error::DataFusionError,
    logical_expr::{ColumnarValue, ScalarFunctionImplementation, ScalarUDF, Volatility},
    sql::sqlparser::parser::ParserError,
};

use crate::datafusion::udf::string_arg::{
    coerce_string_args, create_coercible_udf, int64_param, string_param, with_string_arrays,
};

/// Implementation of fuzzy_match
pub static FUZZY_MATCH_UDF: Lazy<ScalarUDF> = Lazy::new(|| {
    create_coercible_udf(
        crate::datafusion::udf::FUZZY_MATCH_UDF_NAME,
        vec![string_param(), string_param(), int64_param()],
        // returns boolean
        DataType::Boolean,
        Volatility::Stable,
        fuzzy_match_expr_impl(),
    )
});

/// fuzzy_match function for datafusion
pub fn fuzzy_match_expr_impl() -> ScalarFunctionImplementation {
    Arc::new(move |args: &[ColumnarValue]| {
        if args.len() != 3 {
            return Err(DataFusionError::SQL(
                Box::new(ParserError::ParserError(
                    "fuzzy_match UDF expects two string".to_string(),
                )),
                None,
            ));
        }
        let (string_type, args) = coerce_string_args(args, &[0, 1])?;
        let args = ColumnarValue::values_to_arrays(&args)?;
        let (haystack, needle) = (&args[0], &args[1]);
        let distance = as_int64_array(&args[2])?;
        let array = with_string_arrays!(
            string_type,
            [haystack, needle],
            Ok(fuzzy_match_array(haystack, needle, distance))
        )?;
        Ok(ColumnarValue::from(Arc::new(array) as ArrayRef))
    })
}

fn fuzzy_match_array<'a, S: StringArrayType<'a>>(
    haystack: S,
    needle: S,
    distance: &Int64Array,
) -> BooleanArray {
    zip(haystack.iter(), zip(needle.iter(), distance.iter()))
        .map(
            |(haystack, (needle, distance))| match (haystack, needle, distance) {
                // NULL when either haystack or needle is NULL
                (Some(haystack), Some(needle), Some(dis)) => Some(
                    o2_collect_search_tokens(haystack)
                        .iter()
                        .any(|term| levenshtein(term, needle) as i64 <= dis),
                ),
                _ => None,
            },
        )
        .collect()
}

#[cfg(test)]
mod tests {
    use arrow::array::{BooleanArray, StringArray};
    use datafusion::{
        arrow::{
            array::Int64Array,
            datatypes::{Field, Schema},
            record_batch::RecordBatch,
        },
        datasource::MemTable,
        prelude::SessionContext,
    };

    use super::*;

    #[test]
    fn test_fuzzy_match_impl_direct_match() {
        let f = fuzzy_match_expr_impl();
        let haystack = StringArray::from(vec!["hello world"]);
        let needle = StringArray::from(vec!["helo"]);
        let distance = Int64Array::from(vec![1i64]);
        let args = [
            ColumnarValue::Array(Arc::new(haystack)),
            ColumnarValue::Array(Arc::new(needle)),
            ColumnarValue::Array(Arc::new(distance)),
        ];
        let result = f(&args).unwrap();
        if let ColumnarValue::Array(out) = result {
            let out = out.as_any().downcast_ref::<BooleanArray>().unwrap();
            assert!(out.value(0));
        } else {
            panic!("expected array result");
        }
    }

    #[test]
    fn test_fuzzy_match_impl_no_match() {
        let f = fuzzy_match_expr_impl();
        let haystack = StringArray::from(vec!["hello"]);
        let needle = StringArray::from(vec!["xyz"]);
        let distance = Int64Array::from(vec![1i64]);
        let args = [
            ColumnarValue::Array(Arc::new(haystack)),
            ColumnarValue::Array(Arc::new(needle)),
            ColumnarValue::Array(Arc::new(distance)),
        ];
        let result = f(&args).unwrap();
        if let ColumnarValue::Array(out) = result {
            let out = out.as_any().downcast_ref::<BooleanArray>().unwrap();
            assert!(!out.value(0));
        } else {
            panic!("expected array result");
        }
    }

    #[test]
    fn test_fuzzy_match_impl_wrong_arg_count_errors() {
        let f = fuzzy_match_expr_impl();
        let haystack = StringArray::from(vec!["hello"]);
        let args = [ColumnarValue::Array(Arc::new(haystack))];
        assert!(f(&args).is_err());
    }

    #[tokio::test]
    async fn test_fuzzy_match_udf() {
        let sql = "select * from t where fuzzy_match(log, 'data', 1)";

        // define a schema.
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("log", DataType::Utf8, false),
        ]));

        // define data.
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5])),
                Arc::new(StringArray::from(vec![
                    "here has data",
                    "here has date",
                    "here no datt",
                    "here no daxx",
                    "",
                ])),
            ],
        )
        .unwrap();

        let ctx = SessionContext::new();
        ctx.register_udf(FUZZY_MATCH_UDF.clone());

        let provider = MemTable::try_new(schema, vec![vec![batch]]).unwrap();
        ctx.register_table("t", Arc::new(provider)).unwrap();

        let df = ctx.sql(sql).await.unwrap();
        let result = df.collect().await.unwrap();
        let count = result.iter().map(|batch| batch.num_rows()).sum::<usize>();
        assert_eq!(count, 3);
    }
}
