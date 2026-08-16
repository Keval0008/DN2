css/styles.css
js/export.js
js/interact.js
js/layout.js
js/main.js
js/model.js
js/render.js
index.html
render_graph.py


File	Role
index.html	The app — open it directly, drop your JSON on it
render_graph.py	Bakes a JSON + the app into one shareable file
js/model.js	Parse, validate, build the effective graph
js/layout.js	Layered top-down DAG layout
js/render.js	SVG construction
js/interact.js	Zoom/pan, lineage, panel, search
js/export.js	SVG + PNG download
dist/base_data.html	Your sample, baked into one 79 KB file



# open index.html, then drag sample.json onto the page
python render_graph.py sample.json -o dist/base_data.html   # one-file artifact
python render_graph.py --serve 8000                         # dev server; auto-loads sample.json
