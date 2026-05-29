from __future__ import annotations

from ..base_components import BaseTableAppGlue


class TableAppGlue(BaseTableAppGlue):
    """Mediator: sync ``TableWidget`` callbacks with ``App`` navigation/state."""

    def table_select_ctx(self, ctx):
        if ctx["row"] is None:
            return
        self.app.select([ctx["row"]])

    def table_move_ctx(self, ctx):
        if ctx["key"] == "Up":
            self.app.p_decrease()
        elif ctx["key"] == "Down":
            self.app.p_increase()

    def table_commit_ctx(self, ctx):
        row = ctx.get("row")
        col = ctx.get("col")
        val = ctx.get("value")
        if row is None:
            return

        i, j = self.app.to_i_j[row]
        idx = self.app.to_index[i][j]
        if col == 1:
            if self.app.triple_active:
                self.app.delta[idx][1] = val
            df_carto = self.app.carto.cont[i][0]
            df_carto.iat[j, df_carto.columns.get_loc("label_color")] = val

        tree = self.app.table.tree

        def follow_table_final_row():
            cur_iid = tree.cur_iid or (tree.selection()[0] if tree.selection() else None)
            if not cur_iid:
                return
            new_row = tree._row_index_from_iid(cur_iid)
            if new_row is None:
                return
            self.app.i, self.app.j = self.app.to_i_j[new_row]
            self.app.update_plot()

        tree.after_idle(follow_table_final_row)
