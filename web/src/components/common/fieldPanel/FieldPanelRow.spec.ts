// Copyright 2026 OpenObserve Inc.

import { afterEach, describe, expect, it, vi } from "vitest";
import { mount, type VueWrapper } from "@vue/test-utils";
import { defineComponent, h } from "vue";
import FieldPanelRow from "./FieldPanelRow.vue";
import type { PanelField } from "./fieldPanel.types";

const FIELD: PanelField = {
  name: "service_name",
  dataType: "Utf8",
  ftsKey: false,
  isSchemaField: true,
  showValues: true,
};

function mountRow(onRowClick: () => void, armDrag: () => void) {
  const Host = defineComponent({
    setup: () => () =>
      h("div", { "data-test": "host-row", onClick: onRowClick }, [
        h(FieldPanelRow, {
          field: FIELD,
          actions: [],
          timestampColumn: "_timestamp",
          expandable: true,
          drag: true,
          dragEnabled: true,
          armDrag,
        }),
      ]),
  });
  return mount(Host, { attachTo: document.body });
}

describe("FieldPanelRow", () => {
  let wrapper: VueWrapper | undefined;

  afterEach(() => {
    wrapper?.unmount();
    wrapper = undefined;
  });

  it("keeps a drag grip click from toggling the row, while its mousedown still arms the drag", async () => {
    const onRowClick = vi.fn();
    const armDrag = vi.fn();
    wrapper = mountRow(onRowClick, armDrag);
    const grip = wrapper.find('[data-test="o-field-list-drag-indicator"]');

    await grip.trigger("mousedown");
    await grip.trigger("click");

    expect(armDrag).toHaveBeenCalledTimes(1);
    expect(onRowClick).not.toHaveBeenCalled();
  });

  it("lets a click on the field label reach the row so it still toggles", async () => {
    const onRowClick = vi.fn();
    wrapper = mountRow(onRowClick, vi.fn());

    await wrapper.find('[data-test="logs-field-list-item-service_name"]').trigger("click");

    expect(onRowClick).toHaveBeenCalledTimes(1);
  });
});
