# Stelly Explorer

A desktop window onto a [Stelly](https://github.com/nab-os/stelly) server's
space: every analysed track as a point, in the server's own map or in
projections made here, flat or in 3D, coloured by genre, key, tempo or any
column of the space.

Native, drawn by the GPU through egui and wgpu: the whole catalogue
(150k tracks and more) turns and zooms at the display's frame rate.

## What it does

| | |
|---|---|
| **Projections** | **Map**, the server's UMAP of the whole catalogue · **PCA** in 2D or 3D · **Axes**, any two or three columns of the space, raw or spread evenly by rank · **UMAP** in 2D or 3D, run here on a sample |
| **Colour** | **Genre**, the eight biggest in checked colourblind-safe hues, the rest grey · **Key** on the Camelot wheel · **Tempo** · any **column**, on a diverging scale |
| **Focus** | Click a genre or a key in the legend to light only those tracks; a key lights the keys it mixes with too. Search by title, artist or album. |
| **Names** | Each genre named where it gathers, in 2D and in 3D, as the Stelly app's map does |
| **A track** | Hover for its name, tempo and key; click for its details, its mood, and its nearest tracks in the space, drawn as lines on the map |
| **Weights** | How much each part of the space counts, for PCA, UMAP and the nearest tracks; the server's defaults to start |

Drag to move (2D) or turn (3D); right-drag or Shift-drag to move in 3D;
scroll to zoom toward the pointer; **F** fits the view, **Esc** clears the pick.

## Running it

```sh
cargo run --release
```

It uses the Stelly app's pairing when the app is paired on the same machine.
Otherwise it asks for the server's address and a token:

```sh
stelly-server pair --name explorer --scope play
```

`STELLY_SERVER` and `STELLY_TOKEN` override both. The server must have
`GET /api/space/vectors`, from Stelly 2.3.0. The space is kept in
`~/.cache/stelly-explorer`, so a reopen asks the server only whether it changed,
and a server that cannot be reached still leaves the last copy to look at.

On Linux, wgpu wants Vulkan or GL drivers; nothing else outside Cargo.

## How fast

On a synthetic catalogue the production size, 156,000 tracks and 81 columns,
32 threads: weighting 21 ms, PCA 47 ms, axes 11 ms, genre names under 10 ms,
UMAP in 3D on a 25,000-track sample 6.5 s. UMAP's neighbour search is
quadratic, which is why it runs on a sample:

```sh
cargo test --release bench -- --ignored --nocapture
```

## Tests

```sh
cargo test --release
STELLY_SERVER=… STELLY_TOKEN=… cargo test --release source -- --ignored
```

The renderer's test draws real points off screen with whatever GPU adapter is
there, and checks what lands where.

## Licence

MIT or Apache-2.0, at your option.
