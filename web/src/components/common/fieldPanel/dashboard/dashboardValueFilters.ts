// Copyright 2026 OpenObserve Inc.

import { buildCondition } from "@/utils/dashboard/dashboardAutoQueryBuilder";
import {
  hasVariable,
  joinPanelListText,
  splitPanelListText,
  type DashboardColumn,
  type DashboardFilterCondition,
  type DashboardFilterGroup,
} from "@/components/filterBar/codecs/dashboardFilterCodec";
import type { FilterConnector } from "@/utils/query/filterBar";
import { splitQuotedString } from "@/utils/queryUtils";
import type { ActiveFilterValues } from "../activeFilterValues";
import { isNullValue, MISSING_VALUE_KEY } from "../valueFilterExpression";
import type { DashboardPanelQuery, DashboardPanelState } from "./dashboardPanelState";

type FilterNode = DashboardFilterCondition | DashboardFilterGroup;

export interface ValueFilterColumn {
  field: string;
  streamAlias?: string;
}

const isGroup = (node: FilterNode): node is DashboardFilterGroup => node.filterType === "group";

const columnOf = (condition: DashboardFilterCondition): DashboardColumn =>
  typeof condition.column === "string" ? { field: condition.column } : condition.column;

const sameColumn = (condition: DashboardFilterCondition, column: ValueFilterColumn) => {
  const own = columnOf(condition);
  return own.field === column.field && (own.streamAlias ?? undefined) === column.streamAlias;
};

const NULL_OPERATORS = ["Is Null", "Is Not Null"];

const NOT_EQUAL = "<>";

// A whole value the dashboard substitutes, bare or single-quoted, as the codec reads an IN text.
const PLACEHOLDER =
  /^'?(\{\{\s*[\w-]+\s*(:\s*[a-zA-Z]+\s*)?\}\}|\$\{\s*[\w-]+\s*(:\s*[a-zA-Z]+\s*)?\}|\$[\w-]+)'?$/;

const leafValues = (condition: DashboardFilterCondition): string[] => {
  if (condition.type === "list") return (condition.values ?? []).map(String);
  if (NULL_OPERATORS.includes(condition.operator ?? "")) return [MISSING_VALUE_KEY];
  if (condition.operator === NOT_EQUAL) return [String(condition.value ?? "")];
  return splitPanelListText(condition.value);
};

// The codec never writes a variable into a list leaf, so only a condition's text can hold one.
const isVariableLeaf = (condition: DashboardFilterCondition) =>
  condition.type !== "list" && leafValues(condition).some((value) => PLACEHOLDER.test(value));

// Only the leaves the values list writes are its own: a list, an IN / NOT IN / <>, or an Is Null / Is Not Null condition, never one on a `$variable`.
const isValueLeaf = (condition: DashboardFilterCondition) =>
  (condition.type === "list" ||
    condition.operator === "IN" ||
    condition.operator === "NOT IN" ||
    condition.operator === NOT_EQUAL ||
    NULL_OPERATORS.includes(condition.operator ?? "")) &&
  !isVariableLeaf(condition);

const isExcludeLeaf = (condition: DashboardFilterCondition) =>
  condition.operator === "NOT IN" ||
  condition.operator === NOT_EQUAL ||
  condition.operator === "Is Not Null";

// A node's connector joins it to the node before it, so the first one's is never read.
const andJoined = (group: DashboardFilterGroup) =>
  group.conditions.every((node, index) => index === 0 || node.logicalOperator !== "OR");

/** Values each column's own IN and NOT IN leaves hold, keyed by the column's row key. */
export const collectValueFilters = (
  filter: DashboardFilterGroup | undefined,
  columns: { key: string; column: ValueFilterColumn }[],
): ActiveFilterValues => {
  const include: Record<string, string[]> = {};
  const exclude: Record<string, string[]> = {};
  for (const { key, column } of columns) {
    for (const leaf of filter ? ownedLeaves(filter, column) : []) {
      const target = isExcludeLeaf(leaf) ? exclude : include;
      // A null-safe exclusion repeats the included null, which is still one picked value.
      target[key] = [...new Set([...(target[key] ?? []), ...leafValues(leaf)])];
    }
  }
  return { include, exclude };
};

const includeLeaf = (column: ValueFilterColumn, values: string[]): DashboardFilterCondition => ({
  type: "list",
  values: [...values],
  column: { ...column },
  operator: null,
  value: null,
  logicalOperator: "AND",
  filterType: "condition",
});

// The SQL builder pastes an IN text holding `$` or `{{` verbatim, so picked literals are SQL-quoted there.
const pickedListText = (values: string[]) =>
  values.some(hasVariable)
    ? values.map((value) => `'${value.replace(/'/g, "''")}'`).join(",")
    : joinPanelListText(values);

// The SQL builder and the codec both re-split a NOT IN text and trim each value, so a value they would change cannot ride in it.
const survivesListText = (value: string) => {
  const text = joinPanelListText([value]);
  const isValue = (parts: string[] | undefined) => parts?.length === 1 && parts[0] === value;
  return isValue(splitQuotedString(text)) && isValue(splitPanelListText(text));
};

const notInLeaf = (column: ValueFilterColumn, values: string[]): DashboardFilterCondition => ({
  type: "condition",
  values: [],
  column: { ...column },
  operator: "NOT IN",
  value: pickedListText(values),
  logicalOperator: "AND",
  filterType: "condition",
});

// The builder drops a blank `<>` as a value not entered yet unless it is marked as a real empty literal.
const notEqualLeaf = (column: ValueFilterColumn, value: string): DashboardFilterCondition => ({
  type: "condition",
  values: [],
  column: { ...column },
  operator: NOT_EQUAL,
  value,
  logicalOperator: "AND",
  filterType: "condition",
  ...(value === "" ? { emptyLiteral: true } : {}),
});

// A text holding a variable is pasted into the SQL verbatim, so only a plain list needs the values it would mangle split out.
const excludeLeaves = (column: ValueFilterColumn, values: string[]): DashboardFilterCondition[] => {
  if (values.some(hasVariable)) return [notInLeaf(column, values)];
  const listed = values.filter(survivesListText);
  return [
    ...(listed.length ? [notInLeaf(column, listed)] : []),
    ...values
      .filter((value) => !survivesListText(value))
      .map((value) => notEqualLeaf(column, value)),
  ];
};

const nullLeaf = (
  column: ValueFilterColumn,
  operator: "Is Null" | "Is Not Null",
  logicalOperator: FilterConnector,
): DashboardFilterCondition => ({
  type: "condition",
  values: [],
  column: { ...column },
  operator,
  value: null,
  logicalOperator,
  filterType: "condition",
});

// A row never equals the text 'null', so a picked null is a null test; with other values it is ORed in its own group so the flat AND chain keeps its meaning.
const includeNodes = (column: ValueFilterColumn, include: string[]): FilterNode[] => {
  const values = include.filter((value) => !isNullValue(value));
  if (values.length === include.length) return values.length ? [includeLeaf(column, values)] : [];
  if (!values.length) return [nullLeaf(column, "Is Null", "AND")];
  return [
    {
      filterType: "group",
      logicalOperator: "AND",
      conditions: [includeLeaf(column, values), nullLeaf(column, "Is Null", "OR")],
    },
  ];
};

// `f NOT IN (...)` is unknown on a null row, so with null included the exclusion lets null through.
const excludeNodes = (
  column: ValueFilterColumn,
  exclude: string[],
  nullIncluded: boolean,
): FilterNode[] => {
  const values = exclude.filter((value) => !isNullValue(value));
  const leaves = values.length ? excludeLeaves(column, values) : [];
  const notIn: FilterNode[] =
    leaves.length && nullIncluded
      ? [
          {
            filterType: "group",
            logicalOperator: "AND",
            conditions: [...leaves, nullLeaf(column, "Is Null", "OR")],
          },
        ]
      : leaves;
  return [
    ...notIn,
    ...(values.length < exclude.length ? [nullLeaf(column, "Is Not Null", "AND")] : []),
  ];
};

// Included and excluded values sit side by side, the excludes ANDed onto the includes.
const valueNodes = (
  column: ValueFilterColumn,
  include: string[],
  exclude: string[],
): FilterNode[] => [
  ...includeNodes(column, include),
  ...excludeNodes(column, exclude, include.some(isNullValue)),
];

const ownsLeaf = (column: ValueFilterColumn, node: FilterNode) =>
  !isGroup(node) && isValueLeaf(node) && sameColumn(node, column);

// A group of nothing but the column's value leaves is the one written for an included null, so it is owned whole.
const ownedBy = (column: ValueFilterColumn) => (node: FilterNode) =>
  ownsLeaf(column, node) ||
  (isGroup(node) &&
    node.conditions.length > 0 &&
    node.conditions.every((child) => ownsLeaf(column, child)));

// Rewriting a leaf ORed with another field's condition would change what that branch matches, so only AND-joined groups are searched.
const ownedLeaves = (
  group: DashboardFilterGroup,
  column: ValueFilterColumn,
): DashboardFilterCondition[] => {
  if (!andJoined(group)) return [];
  return group.conditions.flatMap((node): DashboardFilterCondition[] => {
    if (!isGroup(node)) return ownsLeaf(column, node) ? [node] : [];
    if (ownedBy(column)(node)) {
      return node.conditions.filter((child): child is DashboardFilterCondition => !isGroup(child));
    }
    return ownedLeaves(node, column);
  });
};

// Depth-first, so the first match is the one a reader of the filter sees first.
const findOwned = (
  group: DashboardFilterGroup,
  owned: (node: FilterNode) => boolean,
): { parent: DashboardFilterGroup; index: number } | null => {
  if (!andJoined(group)) return null;
  for (const [index, node] of group.conditions.entries()) {
    if (owned(node)) return { parent: group, index };
    if (isGroup(node)) {
      const found = findOwned(node, owned);
      if (found) return found;
    }
  }
  return null;
};

// Ownership is decided before a group is touched, so an owned group goes whole; a group left empty by the drop goes too.
const dropOwned = (
  group: DashboardFilterGroup,
  owned: (node: FilterNode) => boolean,
  keep: FilterNode | null,
) => {
  if (!andJoined(group)) return;
  group.conditions = group.conditions.filter((node) => {
    if (node === keep) return true;
    if (owned(node)) return false;
    if (!isGroup(node) || !node.conditions.length) return true;
    dropOwned(node, owned, keep);
    return node.conditions.length > 0;
  });
};

/** Replaces the column's own value leaves with one for the included and one for the excluded values, where the first of them sat in the filter tree. */
export const writeValueFilter = (
  query: DashboardPanelQuery,
  column: ValueFilterColumn,
  include: string[],
  exclude: string[],
) => {
  if (!query.fields.filter) {
    query.fields.filter = { filterType: "group", logicalOperator: "AND", conditions: [] };
  }
  const root = query.fields.filter;
  const owned = ownedBy(column);
  const at = findOwned(root, owned);
  const nodes = valueNodes(column, include, exclude);
  if (!nodes.length) return;
  if (!at) {
    // Appended to an OR chain the new leaves would bind only to its last term, so the chain is grouped first.
    if (!andJoined(root)) {
      root.conditions = [
        { filterType: "group", logicalOperator: "AND", conditions: root.conditions },
      ];
    }
    root.conditions.push(...nodes);
    return;
  }
  // The surviving node is the one read back from the tree, so it compares equal through a reactive proxy.
  const first = at.parent.conditions[at.index];
  nodes[0].logicalOperator = first.logicalOperator;
  dropOwned(root, owned, first);
  at.parent.conditions.splice(at.parent.conditions.indexOf(first), 1, ...nodes);
};

export const removeValueFilter = (query: DashboardPanelQuery, column: ValueFilterColumn) => {
  if (query.fields.filter) dropOwned(query.fields.filter, ownedBy(column), null);
};

const mentionsVariable = (condition: DashboardFilterCondition) =>
  [condition.value, ...(condition.values ?? [])].some(
    (value) => typeof value === "string" && hasVariable(value),
  );

// A field's own conditions would narrow its counts to the picked values; a `$variable` is only substituted when the panel runs; null drops the node.
const pruneFilter = (node: FilterNode, fieldName: string): FilterNode | null => {
  if (!isGroup(node)) {
    return columnOf(node).field === fieldName || mentionsVariable(node) ? null : node;
  }
  const pruned = node.conditions.map((child) => pruneFilter(child, fieldName));
  if (pruned.every((child, index) => child === node.conditions[index])) return node;
  // Dropping one term of an OR would narrow it, so the whole OR goes and only widens the values.
  if (!andJoined(node)) return null;
  const conditions = pruned.filter((child): child is FilterNode => child !== null);
  return conditions.length ? { ...node, conditions } : null;
};

/** The builder filter as a WHERE fragment for one field's values query, or "" when it has none. */
export const builderWhereFor = (panel: DashboardPanelState, fieldName: string): string => {
  const filter = panel.data.queries[panel.layout.currentQueryIndex]?.fields?.filter;
  if (!filter?.conditions?.length) return "";
  const pruned = pruneFilter(filter, fieldName);
  return pruned ? buildCondition(pruned, panel) : "";
};
