# ANTIPOMPEII
**A N**ested **T**ool for **I**ntegrated **P**lanning **O**f **M**ulti-hazard **P**reparedness and **E**mergency **I**nfrastructure **I**ntervention

ANTIPOMPEII is an interactive CLI for urban vulnerability assessment and resilience analysis. It downloads OpenStreetMap street networks and buildings, enriches them with population demographics, DEM elevation, and hazard disruption layers, builds a `graph-tool` network, runs structural analytics, and produces LLM-augmented interpretation reports.

Each interactive run targets one **case study** identified by `(date, location)`.

## Features

- **Automated data collection** — OSM streets and 7 thematic building layers via `osmnx`, WorldPop demographics, DEM from OpenTopography (requires their token; one can create it for free), and an OSM water layer (rivers, lakes, wetlands, coastline).
- **Network construction** — population, elevation, distance-to-water, facility, and disruption attributes attached to every edge.
- **Analytics modules**
  - `StatsAnalyst` — direct and indirect disruption tables (roads, facilities, population) for _ex-post_ assessment of the disaster.
  - `RobustnessEstimator` — graph metrics relevant to urban robustness across disruption states for _ex-ante_ system-wide assessments.
  - `Percolator` — progressive edge-removal under betweenness attack, random failure, or elevation flood (_ex-ante_).
  - `VulnerabilitySimulator` — vulnerability map creation and assessment (_ex-ante_).
- **Temporal mode** — every pipeline stage iterates over a list of snapshots, so a single run can compare multiple timestamps of the same location.
- **Cross-case comparison** — multiple existing graphs are loaded and side-by-side analytics is produced.
- **LLM interpretation** — per-module narratives and a cross-module executive summary via `litellm`.

## Installation

This project depends on [`graph-tool`](https://figshare.com/articles/dataset/graph_tool/1164194?file=8540740) among other Python modules. Dependency management uses [pixi](https://pixi.sh).

```bash
git clone https://github.com/kiparisus/antipompeii.git
cd antipompeii
pixi install
```

## Quick start

```bash
pixi run start
```

The CLI walks you through:

1. Selecting an **operational mode** (see below).
2. Choosing a **city** and **extent** (bounding box, place name, or polygon).
3. Picking one or more **timestamps** (single or temporal run).
4. Running the enrichment pipeline and analytics, optionally followed by an LLM interpretation pass.

## Configuration

Setting up `src/antipompeii/config.yaml` enables non-interactive runs.

## Building layer taxonomy

OSM buildings are classified into 7 thematic layers currently across 3 mutually-included spheres (_socio-_, _techno-_, and _orgsphere_; _infosphere_ to be added).

## Author

Pavel Kiparisov — `pavel@kiparisov.space`

## Background papers

[1] Kiparisov, P., Lagutov, V., & Pflug, G. (2023). Quantification of loss of access to critical services during floods in Greater Jakarta: integrating social, geospatial, and network Perspectives. Remote Sensing, 15(21), 5250. https://doi.org/10.3390/rs15215250

[2] Kiparisov, P., Lagutov, V. (2024). Integrated GIS- and network-based framework for assessing urban critical infrastructure accessibility and resilience: the case of Hurricane Michael. https://arxiv.org/abs/2412.13728

## Note

The complete methodological pipeline of this tool has not yet undergone peer review and has not been validated in real-world settings; therefore, its outputs should be used for informational purposes only.
