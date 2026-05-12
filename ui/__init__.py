"""Interactive UI for the OFC AI.

Subpackages
-----------
- `state_builder` : build a `GameState` from a JSON-shaped spec.
- `analyzer`      : run the AI on a state and return stats.
- `server`        : tiny stdlib HTTP server that exposes both via JSON
                    and serves the static HTML/JS page.

Run:
    python -m ui.server --run artifacts/run_<TS>/ --port 8000
Then open http://localhost:8000/ in a browser.
"""
