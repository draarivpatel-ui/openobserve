// Copyright 2026 OpenObserve Inc.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

use std::sync::Arc;

use crate::{
    config::{Config, SIZE_IN_MB, get_config},
    meta::cluster::Role,
    utils::size::bytes_to_human_readable,
};

/// Every memory consumer whose size is derived from node memory; sizes are read only from here.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemoryConsumer {
    MemTable,
    DataFusionPool,
    FileMemoryCache,
    FileStatCache,
    MetricsBlocksCache,
    IndexFooterCache,
    LabelCache,
    BloomFooterCache,
    MetricsResultCache,
}

impl MemoryConsumer {
    pub const ALL: [MemoryConsumer; 9] = [
        MemoryConsumer::MemTable,
        MemoryConsumer::DataFusionPool,
        MemoryConsumer::FileMemoryCache,
        MemoryConsumer::FileStatCache,
        MemoryConsumer::MetricsBlocksCache,
        MemoryConsumer::IndexFooterCache,
        MemoryConsumer::LabelCache,
        MemoryConsumer::BloomFooterCache,
        MemoryConsumer::MetricsResultCache,
    ];

    pub fn spec(self) -> ConsumerSpec {
        match self {
            MemoryConsumer::MemTable => ConsumerSpec {
                name: "memtable",
                default_ratio: DefaultRatio::Fixed(80),
                enabled_on: EnabledOn::Ingester,
                scope: Scope::Process,
                ratio_env: "ZO_MEMORY_RATIO_MEMTABLE",
                legacy_env: "ZO_MEM_TABLE_MAX_SIZE",
            },
            MemoryConsumer::DataFusionPool => ConsumerSpec {
                name: "datafusion_pool",
                default_ratio: DefaultRatio::ByIngester {
                    ingester: 50,
                    other: 80,
                },
                enabled_on: EnabledOn::QueryOrMerge,
                scope: Scope::PerPool,
                ratio_env: "ZO_MEMORY_RATIO_DATAFUSION_POOL",
                legacy_env: "ZO_MEMORY_CACHE_DATAFUSION_MAX_SIZE",
            },
            MemoryConsumer::FileMemoryCache => ConsumerSpec {
                name: "file_memory_cache",
                default_ratio: DefaultRatio::Fixed(50),
                enabled_on: EnabledOn::FileMemoryCacheSwitch,
                scope: Scope::Process,
                ratio_env: "ZO_MEMORY_RATIO_FILE_MEMORY_CACHE",
                legacy_env: "ZO_MEMORY_CACHE_MAX_SIZE",
            },
            MemoryConsumer::FileStatCache => ConsumerSpec {
                name: "file_stat_cache",
                default_ratio: DefaultRatio::Fixed(5),
                enabled_on: EnabledOn::Querier,
                scope: Scope::Process,
                ratio_env: "ZO_MEMORY_RATIO_FILE_STAT_CACHE",
                legacy_env: "ZO_DATAFUSION_FILE_STAT_CACHE_MAX_SIZE",
            },
            MemoryConsumer::MetricsBlocksCache => ConsumerSpec {
                name: "metrics_blocks_cache",
                default_ratio: DefaultRatio::Fixed(5),
                enabled_on: EnabledOn::Querier,
                scope: Scope::Process,
                ratio_env: "ZO_MEMORY_RATIO_METRICS_BLOCKS_CACHE",
                legacy_env: "ZO_METRICS_BLOCKS_CACHE_MAX_SIZE",
            },
            MemoryConsumer::IndexFooterCache => ConsumerSpec {
                name: "index_footer_cache",
                default_ratio: DefaultRatio::Fixed(5),
                enabled_on: EnabledOn::QuerierOrCompactor,
                scope: Scope::Process,
                ratio_env: "ZO_MEMORY_RATIO_INDEX_FOOTER_CACHE",
                legacy_env: "ZO_INVERTED_INDEX_FOOTER_CACHE_MAX_SIZE",
            },
            MemoryConsumer::LabelCache => ConsumerSpec {
                name: "label_cache",
                default_ratio: DefaultRatio::Fixed(5),
                enabled_on: EnabledOn::Querier,
                scope: Scope::Process,
                ratio_env: "ZO_MEMORY_RATIO_LABEL_CACHE",
                legacy_env: "ZO_METRICS_LABEL_CACHE_MAX_SIZE",
            },
            MemoryConsumer::BloomFooterCache => ConsumerSpec {
                name: "bloom_footer_cache",
                default_ratio: DefaultRatio::Fixed(1),
                enabled_on: EnabledOn::Querier,
                scope: Scope::Process,
                ratio_env: "ZO_MEMORY_RATIO_BLOOM_FOOTER_CACHE",
                legacy_env: "ZO_BLOOM_FOOTER_CACHE_MAX_SIZE",
            },
            MemoryConsumer::MetricsResultCache => ConsumerSpec {
                name: "metrics_result_cache",
                default_ratio: DefaultRatio::Fixed(1),
                enabled_on: EnabledOn::QuerierOrScheduler,
                scope: Scope::Process,
                ratio_env: "ZO_MEMORY_RATIO_METRICS_RESULT_CACHE",
                legacy_env: "ZO_METRICS_RESULT_CACHE_MAX_SIZE",
            },
        }
    }

    pub fn name(self) -> &'static str {
        self.spec().name
    }
}

/// How one consumer is registered; ratios and `ratio_env` are not read until the ratio model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConsumerSpec {
    pub name: &'static str,
    pub default_ratio: DefaultRatio,
    pub enabled_on: EnabledOn,
    pub scope: Scope,
    pub ratio_env: &'static str,
    /// The MB setting the consumer reads today.
    pub legacy_env: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DefaultRatio {
    Fixed(u32),
    ByIngester { ingester: u32, other: u32 },
}

/// Which nodes can fill the consumer, from the code paths that insert into it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnabledOn {
    Ingester,
    Querier,
    /// The compactor reads tantivy footers when it builds bloom filters after a merge.
    QuerierOrCompactor,
    /// The scheduler leads PromQL alert, recording rule and SLO queries.
    QuerierOrScheduler,
    /// Queries, search leaders and merges each build a pool.
    QueryOrMerge,
    /// `ZO_MEMORY_CACHE_ENABLED`, on the nodes that lead or serve searches.
    FileMemoryCacheSwitch,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    Process,
    /// Every query or merge builds its own pool of the quota size.
    PerPool,
}

/// The roles of a node, as the consumers' enablement needs them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NodeRoles {
    pub ingester: bool,
    pub querier: bool,
    pub compactor: bool,
    pub scheduler: bool,
    pub compactor_only: bool,
}

impl NodeRoles {
    pub fn new(roles: &[Role]) -> Self {
        let all = roles.contains(&Role::All);
        Self {
            ingester: all || roles.contains(&Role::Ingester),
            querier: all || roles.contains(&Role::Querier),
            compactor: all || roles.contains(&Role::Compactor),
            scheduler: all || roles.contains(&Role::Scheduler),
            compactor_only: roles == [Role::Compactor],
        }
    }

    fn parse(node_role: &str) -> Self {
        // check_memory_config already panics on an invalid role, so none is dropped here
        let roles: Vec<Role> = node_role
            .split(',')
            .filter_map(|role| role.parse().ok())
            .collect();
        Self::new(&roles)
    }
}

/// Quotas of every consumer, computed from a resolved config.
pub struct MemoryBudget {
    cfg: Arc<Config>,
}

impl MemoryBudget {
    pub fn new(cfg: Arc<Config>) -> Self {
        Self { cfg }
    }

    /// Bytes the consumer may hold; for a `PerPool` consumer this is the size of one pool.
    pub fn quota(&self, consumer: MemoryConsumer) -> usize {
        let memory = &self.cfg.memory;
        match consumer {
            MemoryConsumer::MemTable => memory.mem_table_max_size,
            MemoryConsumer::DataFusionPool => memory.datafusion_pool_size,
            MemoryConsumer::FileMemoryCache => {
                memory.file_cache_max_size * memory.file_cache_bucket_num
            }
            MemoryConsumer::FileStatCache => memory.datafusion_file_stat_cache_max_size,
            MemoryConsumer::MetricsBlocksCache => memory
                .metrics_blocks_cache_max_size
                .saturating_mul(SIZE_IN_MB as usize),
            MemoryConsumer::IndexFooterCache => memory.inverted_index_footer_cache_max_size,
            MemoryConsumer::LabelCache => memory.metrics_label_cache_max_size,
            MemoryConsumer::BloomFooterCache => memory.bloom_footer_cache_max_size,
            MemoryConsumer::MetricsResultCache => memory.metrics_result_cache_max_size,
        }
    }

    pub fn is_enabled(&self, consumer: MemoryConsumer) -> bool {
        let roles = self.roles();
        let on_role = match consumer.spec().enabled_on {
            EnabledOn::Ingester => roles.ingester,
            EnabledOn::Querier => roles.querier,
            EnabledOn::QuerierOrCompactor => roles.querier || roles.compactor,
            EnabledOn::QuerierOrScheduler => roles.querier || roles.scheduler,
            EnabledOn::QueryOrMerge => {
                roles.ingester || roles.querier || roles.compactor || roles.scheduler
            }
            EnabledOn::FileMemoryCacheSwitch => {
                self.cfg.memory.file_cache_enabled && (roles.querier || roles.scheduler)
            }
        };
        on_role && self.quota(consumer) > 0
    }

    /// Pools of this consumer that can be live at once: one per merge thread on a compactor-only
    /// node, whose pool size is split across them.
    pub fn pools(&self, consumer: MemoryConsumer) -> usize {
        match consumer.spec().scope {
            Scope::PerPool if self.roles().compactor_only => self.cfg.limit.file_merge_thread_num,
            Scope::PerPool | Scope::Process => 1,
        }
    }

    /// Bytes the consumer counts toward the node total; zero when it is off on this node.
    pub fn limit(&self, consumer: MemoryConsumer) -> usize {
        if self.is_enabled(consumer) {
            self.quota(consumer) * self.pools(consumer)
        } else {
            0
        }
    }

    pub fn total(&self) -> usize {
        MemoryConsumer::ALL.iter().map(|c| self.limit(*c)).sum()
    }

    /// The startup allocation list; `env_set` tells whether a consumer's legacy env is set.
    pub fn report(&self, env_set: impl Fn(&str) -> bool) -> String {
        let mem_total = self.cfg.limit.mem_total;
        let total = self.total();
        let mut lines = vec![
            format!(
                "Memory budget: node memory {}, roles: {}",
                bytes_to_human_readable(mem_total as f64),
                self.cfg.common.node_role
            ),
            format!(
                "  {:<22} {:<8} {:<8} {:>12}  scope",
                "item", "enabled", "source", "quota"
            ),
        ];
        for consumer in MemoryConsumer::ALL {
            let spec = consumer.spec();
            let enabled = self.is_enabled(consumer);
            let quota = if enabled {
                bytes_to_human_readable(self.quota(consumer) as f64)
            } else {
                "-".to_string()
            };
            let scope = match (spec.scope, self.pools(consumer)) {
                (Scope::Process, _) => "process".to_string(),
                (Scope::PerPool, 1) => "per pool".to_string(),
                (Scope::PerPool, pools) => format!("per pool x{pools}"),
            };
            lines.push(format!(
                "  {:<22} {:<8} {:<8} {:>12}  {scope}",
                spec.name,
                if enabled { "yes" } else { "no" },
                if env_set(spec.legacy_env) {
                    "env"
                } else {
                    "default"
                },
                quota,
            ));
        }
        let percent = match (total as u128 * 100).checked_div(mem_total as u128) {
            Some(percent) => format!("{percent}%"),
            None => "-".to_string(),
        };
        lines.push(format!(
            "  {:<22} {:<17} {:>12}  {percent} of node memory",
            "total",
            "",
            bytes_to_human_readable(total as f64),
        ));
        lines.join("\n")
    }

    fn roles(&self) -> NodeRoles {
        NodeRoles::parse(&self.cfg.common.node_role)
    }
}

pub fn memory_budget() -> MemoryBudget {
    MemoryBudget::new(get_config())
}

/// True when the env var holds a non-zero value, which is what makes a legacy MB setting apply.
pub fn env_is_set(name: &str) -> bool {
    std::env::var(name).is_ok_and(|value| {
        let value = value.trim();
        !value.is_empty() && value != "0"
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::resolve_memory_config;

    const MB: usize = 1024 * 1024;
    const GB: usize = 1024 * MB;
    const CPU_NUM: usize = 4;
    const MERGE_THREADS: usize = 2;
    const SIZES: [usize; 4] = [512 * MB, 2 * GB, 16 * GB, 64 * GB];

    fn config(node_role: &str, local_mode: bool) -> Config {
        let mut cfg = Config::default();
        cfg.common.node_role = node_role.to_string();
        cfg.common.local_mode = local_mode;
        cfg.limit.cpu_num = CPU_NUM;
        cfg.limit.file_merge_thread_num = MERGE_THREADS;
        cfg
    }

    fn resolve(mut cfg: Config, mem_total: usize) -> MemoryBudget {
        resolve_memory_config(&mut cfg, mem_total).unwrap();
        MemoryBudget::new(Arc::new(cfg))
    }

    fn budget(node_role: &str, mem_total: usize) -> MemoryBudget {
        resolve(config(node_role, false), mem_total)
    }

    fn percent_mb(mem_total: usize, percent: f64, min_mb: usize, max_mb: usize) -> usize {
        ((mem_total as f64 / MB as f64 * percent) as usize).clamp(min_mb, max_mb) * MB
    }

    /// Today's defaults, written out from `check_memory_config`.
    fn expected_quota(
        consumer: MemoryConsumer,
        node_role: &str,
        local_mode: bool,
        mem: usize,
    ) -> usize {
        let file_cache = if local_mode { mem / 4 } else { mem / 2 };
        match consumer {
            MemoryConsumer::MemTable => file_cache,
            MemoryConsumer::DataFusionPool if node_role == "compactor" => mem / MERGE_THREADS,
            MemoryConsumer::DataFusionPool if local_mode => (mem - file_cache) / 2,
            MemoryConsumer::DataFusionPool => mem - file_cache,
            MemoryConsumer::FileMemoryCache => file_cache / CPU_NUM * CPU_NUM,
            MemoryConsumer::FileStatCache | MemoryConsumer::IndexFooterCache => {
                percent_mb(mem, 0.05, 8, 1024)
            }
            MemoryConsumer::MetricsBlocksCache => (mem / MB / 20).clamp(128, 4096) * MB,
            MemoryConsumer::LabelCache => (mem * 5 / 100).clamp(8 * MB, GB),
            MemoryConsumer::BloomFooterCache => percent_mb(mem, 0.01, 4, 256),
            MemoryConsumer::MetricsResultCache => percent_mb(mem, 0.01, 32, 256),
        }
    }

    fn enabled(budget: &MemoryBudget) -> Vec<MemoryConsumer> {
        MemoryConsumer::ALL
            .into_iter()
            .filter(|c| budget.is_enabled(*c))
            .collect()
    }

    #[test]
    fn test_quotas_match_check_memory_config_defaults() {
        let roles = [
            ("querier", false),
            ("ingester", false),
            ("compactor", false),
            ("router", false),
            ("alertmanager", false),
            ("ingester,querier", false),
            ("querier,compactor", false),
            ("all", false),
            ("all", true),
        ];
        for mem in SIZES {
            for (node_role, local_mode) in roles {
                let budget = resolve(config(node_role, local_mode), mem);
                for consumer in MemoryConsumer::ALL {
                    assert_eq!(
                        budget.quota(consumer),
                        expected_quota(consumer, node_role, local_mode, mem),
                        "{consumer:?} on {node_role} (local {local_mode}) with {mem} bytes"
                    );
                }
            }
        }
    }

    #[test]
    fn test_enabled_consumers_by_role() {
        use MemoryConsumer::*;
        let querier = vec![
            DataFusionPool,
            FileStatCache,
            MetricsBlocksCache,
            IndexFooterCache,
            LabelCache,
            BloomFooterCache,
            MetricsResultCache,
        ];
        let mut all = vec![MemTable];
        all.extend(&querier);
        let cases: [(&str, Vec<MemoryConsumer>); 8] = [
            ("querier", querier.clone()),
            ("ingester", vec![MemTable, DataFusionPool]),
            ("compactor", vec![DataFusionPool, IndexFooterCache]),
            ("router", vec![]),
            ("alertmanager", vec![DataFusionPool, MetricsResultCache]),
            ("ingester,querier", all.clone()),
            ("querier,compactor", querier.clone()),
            ("all", all.clone()),
        ];
        for (node_role, expected) in cases {
            assert_eq!(
                enabled(&budget(node_role, 16 * GB)),
                expected,
                "{node_role}"
            );
        }
    }

    #[test]
    fn test_file_memory_cache_enabled_by_switch_on_search_nodes() {
        for (node_role, expected) in [
            ("querier", true),
            ("alertmanager", true),
            ("all", true),
            ("ingester", false),
            ("compactor", false),
        ] {
            let mut cfg = config(node_role, false);
            cfg.memory.file_cache_enabled = true;
            let budget = resolve(cfg, 16 * GB);
            assert_eq!(
                budget.is_enabled(MemoryConsumer::FileMemoryCache),
                expected,
                "{node_role}"
            );
        }
        let off = budget("querier", 16 * GB);
        assert!(!off.is_enabled(MemoryConsumer::FileMemoryCache));
        assert_eq!(off.limit(MemoryConsumer::FileMemoryCache), 0);
    }

    #[test]
    fn test_total_counts_every_merge_pool_on_compactor_only_nodes() {
        let compactor = budget("compactor", 16 * GB);
        assert_eq!(
            compactor.pools(MemoryConsumer::DataFusionPool),
            MERGE_THREADS
        );
        assert_eq!(compactor.limit(MemoryConsumer::DataFusionPool), 16 * GB);
        assert_eq!(compactor.total(), 16 * GB + 819 * MB);

        let mixed = budget("querier,compactor", 16 * GB);
        assert_eq!(mixed.pools(MemoryConsumer::DataFusionPool), 1);
        assert_eq!(mixed.pools(MemoryConsumer::IndexFooterCache), 1);
    }

    #[test]
    fn test_total_is_the_sum_of_enabled_limits() {
        for mem in SIZES {
            for node_role in ["querier", "ingester", "compactor", "router", "all"] {
                let budget = budget(node_role, mem);
                let expected: usize = enabled(&budget)
                    .into_iter()
                    .map(|c| budget.quota(c) * budget.pools(c))
                    .sum();
                assert_eq!(budget.total(), expected, "{node_role} with {mem} bytes");
            }
        }
    }

    #[test]
    fn test_explicit_mb_settings_keep_their_meaning() {
        let mut cfg = config("querier", false);
        cfg.memory.mem_table_max_size = 1024;
        cfg.memory.datafusion_pool_size = 2048;
        cfg.memory.metrics_label_cache_max_size = 100;
        cfg.memory.metrics_blocks_cache_max_size = 64;
        cfg.memory.metrics_result_cache_max_size = 10;
        let budget = resolve(cfg, 16 * GB);
        assert_eq!(budget.quota(MemoryConsumer::MemTable), GB);
        assert_eq!(budget.quota(MemoryConsumer::DataFusionPool), 2 * GB);
        assert_eq!(budget.quota(MemoryConsumer::LabelCache), 100 * MB);
        assert_eq!(budget.quota(MemoryConsumer::MetricsBlocksCache), 64 * MB);
        assert_eq!(budget.quota(MemoryConsumer::MetricsResultCache), 32 * MB);

        let mut cfg = config("querier", false);
        cfg.memory.metrics_blocks_cache_max_size = 5;
        let budget = resolve(cfg, 16 * GB);
        assert_eq!(budget.quota(MemoryConsumer::MetricsBlocksCache), 0);
        assert!(!budget.is_enabled(MemoryConsumer::MetricsBlocksCache));
    }

    #[test]
    fn test_report_lists_every_consumer_and_the_total() {
        let report = budget("compactor", 16 * GB).report(|env| env == "ZO_MEM_TABLE_MAX_SIZE");
        let lines: Vec<&str> = report.lines().collect();
        assert_eq!(lines.len(), MemoryConsumer::ALL.len() + 3);
        assert!(lines[0].contains("node memory 16.00 GB, roles: compactor"));
        for (line, consumer) in lines[2..].iter().zip(MemoryConsumer::ALL) {
            assert!(line.trim_start().starts_with(consumer.name()), "{line}");
        }
        let row = |name: &str| *lines.iter().find(|l| l.contains(name)).unwrap();
        assert!(row("memtable").contains(" no "));
        assert!(row("memtable").contains(" env "));
        assert!(row("datafusion_pool").contains("8.00 GB  per pool x2"));
        assert!(row("datafusion_pool").contains(" default "));
        assert!(row("index_footer_cache").contains("819.00 MB  process"));
        assert!(lines[lines.len() - 1].ends_with("16.80 GB  104% of node memory"));

        let querier = budget("querier", 16 * GB).report(|_| false);
        assert!(querier.ends_with("71% of node memory"), "{querier}");
    }

    #[test]
    fn test_consumer_specs() {
        use MemoryConsumer::*;
        let ratio_on = |ratio: DefaultRatio, ingester: bool| match ratio {
            DefaultRatio::Fixed(ratio) => ratio,
            DefaultRatio::ByIngester { ingester: on, .. } if ingester => on,
            DefaultRatio::ByIngester { other, .. } => other,
        };
        let cases = [
            (MemTable, 80, 80, Scope::Process, "ZO_MEM_TABLE_MAX_SIZE"),
            (
                DataFusionPool,
                80,
                50,
                Scope::PerPool,
                "ZO_MEMORY_CACHE_DATAFUSION_MAX_SIZE",
            ),
            (
                FileMemoryCache,
                50,
                50,
                Scope::Process,
                "ZO_MEMORY_CACHE_MAX_SIZE",
            ),
            (
                FileStatCache,
                5,
                5,
                Scope::Process,
                "ZO_DATAFUSION_FILE_STAT_CACHE_MAX_SIZE",
            ),
            (
                MetricsBlocksCache,
                5,
                5,
                Scope::Process,
                "ZO_METRICS_BLOCKS_CACHE_MAX_SIZE",
            ),
            (
                IndexFooterCache,
                5,
                5,
                Scope::Process,
                "ZO_INVERTED_INDEX_FOOTER_CACHE_MAX_SIZE",
            ),
            (
                LabelCache,
                5,
                5,
                Scope::Process,
                "ZO_METRICS_LABEL_CACHE_MAX_SIZE",
            ),
            (
                BloomFooterCache,
                1,
                1,
                Scope::Process,
                "ZO_BLOOM_FOOTER_CACHE_MAX_SIZE",
            ),
            (
                MetricsResultCache,
                1,
                1,
                Scope::Process,
                "ZO_METRICS_RESULT_CACHE_MAX_SIZE",
            ),
        ];
        assert_eq!(cases.len(), MemoryConsumer::ALL.len());
        for ((consumer, on_querier, on_ingester, scope, legacy_env), listed) in
            cases.into_iter().zip(MemoryConsumer::ALL)
        {
            let spec = consumer.spec();
            assert_eq!(consumer, listed);
            assert_eq!(
                ratio_on(spec.default_ratio, false),
                on_querier,
                "{consumer:?}"
            );
            assert_eq!(
                ratio_on(spec.default_ratio, true),
                on_ingester,
                "{consumer:?}"
            );
            assert_eq!(spec.scope, scope, "{consumer:?}");
            assert_eq!(spec.legacy_env, legacy_env, "{consumer:?}");
            assert_eq!(
                spec.ratio_env,
                format!("ZO_MEMORY_RATIO_{}", spec.name.to_uppercase())
            );
        }
    }
}
