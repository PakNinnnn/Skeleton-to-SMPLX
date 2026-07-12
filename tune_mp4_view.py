import argparse
import json
import os

import numpy as np
import trimesh


HTML_TEMPLATE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>SMPL-X MP4 View Tuner</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; }
    #wrap { display: grid; grid-template-columns: minmax(0, 1fr) 360px; height: 100vh; }
    #plot { width: 100%; height: 100%; }
    #panel { border-left: 1px solid #ddd; padding: 16px; overflow: auto; }
    label { display: block; margin-top: 12px; font-weight: 600; }
    pre { white-space: pre-wrap; background: #f6f6f6; padding: 10px; border-radius: 6px; }
    button { margin-top: 10px; padding: 8px 10px; }
    .hint { color: #555; font-size: 13px; line-height: 1.4; }
  </style>
</head>
<body>
<div id="wrap">
  <div id="plot"></div>
  <div id="panel">
    <h2>MP4 View Tuner</h2>
    <p class="hint">
      Drag/rotate/zoom the mesh. The values below update after each camera move.
      Copy the command values into <code>visualize_own.py</code>.
    </p>

    <label>Matplotlib MP4 Command</label>
    <pre id="mplCommand"></pre>
    <button onclick="copyText('mplCommand')">Copy command values</button>

    <label>Plotly Camera</label>
    <pre id="plotlyCamera"></pre>
    <button onclick="copyText('plotlyCamera')">Copy Plotly camera</button>

    <p class="hint">
      Note: <code>elev</code> and <code>azim</code> are approximated from Plotly's camera eye.
      <code>roll</code> is read from the camera up vector when possible. If roll feels off,
      tune <code>--mp4_roll</code> in small steps like 10 or 15 degrees.
    </p>
  </div>
</div>

<script>
const vertices = __VERTICES__;
const faces = __FACES__;
const initialCamera = __CAMERA__;

const trace = {
  type: 'mesh3d',
  x: vertices.map(v => v[0]),
  y: vertices.map(v => v[1]),
  z: vertices.map(v => v[2]),
  i: faces.map(f => f[0]),
  j: faces.map(f => f[1]),
  k: faces.map(f => f[2]),
  color: 'rgb(242, 150, 55)',
  opacity: 1.0,
  flatshading: true
};

const layout = {
  title: 'Drag to Tune MP4 Camera',
  scene: {
    aspectmode: 'data',
    camera: initialCamera,
    xaxis: { title: 'X' },
    yaxis: { title: 'Y' },
    zaxis: { title: 'Z' }
  },
  margin: { l: 0, r: 0, t: 40, b: 0 }
};

function rad2deg(radians) {
  return radians * 180.0 / Math.PI;
}

function cameraToMpl(camera) {
  const eye = camera.eye || {x: 1.25, y: 1.25, z: 1.25};
  const up = camera.up || {x: 0, y: 0, z: 1};
  const xy = Math.sqrt(eye.x * eye.x + eye.y * eye.y);
  const elev = rad2deg(Math.atan2(eye.z, xy));
  const azim = rad2deg(Math.atan2(eye.y, eye.x));
  const roll = rad2deg(Math.atan2(up.x, up.z));
  return {elev, azim, roll};
}

function fmt(value) {
  return Number(value).toFixed(3);
}

function updatePanel(camera) {
  const mpl = cameraToMpl(camera);
  document.getElementById('mplCommand').textContent =
    `--mp4_elev ${fmt(mpl.elev)} \\\\` + "\\n" +
    `--mp4_azim ${fmt(mpl.azim)} \\\\` + "\\n" +
    `--mp4_roll ${fmt(mpl.roll)}`;

  document.getElementById('plotlyCamera').textContent =
    JSON.stringify(camera, null, 2);
}

function currentCameraFromGraph() {
  const gd = document.getElementById('plot');
  return gd.layout.scene.camera;
}

function copyText(id) {
  navigator.clipboard.writeText(document.getElementById(id).textContent);
}

Plotly.newPlot('plot', [trace], layout, {responsive: true}).then(gd => {
  updatePanel(currentCameraFromGraph());
  gd.on('plotly_relayout', eventData => {
    const camera = eventData['scene.camera'] || currentCameraFromGraph();
    updatePanel(camera);
  });
});
</script>
</body>
</html>
"""


def choose_mesh_path(mesh_dir, frame_idx):
    mesh_paths = sorted(
        os.path.join(mesh_dir, name)
        for name in os.listdir(mesh_dir)
        if name.lower().endswith('.obj')
    )
    if not mesh_paths:
        raise FileNotFoundError(f'No .obj meshes found in {mesh_dir}')
    if frame_idx < 0 or frame_idx >= len(mesh_paths):
        raise ValueError(f'frame_idx must be in [0, {len(mesh_paths) - 1}]')
    return mesh_paths[frame_idx]


def camera_from_elev_azim(elev, azim, radius):
    elev_rad = np.deg2rad(elev)
    azim_rad = np.deg2rad(azim)
    xy = radius * np.cos(elev_rad)
    return {
        'eye': {
            'x': float(xy * np.cos(azim_rad)),
            'y': float(xy * np.sin(azim_rad)),
            'z': float(radius * np.sin(elev_rad)),
        },
        'up': {'x': 0, 'y': 0, 'z': 1},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh_dir', required=True)
    parser.add_argument('--frame_idx', type=int, default=0)
    parser.add_argument('--output_html', default='mp4_view_tuner.html')
    parser.add_argument('--initial_elev', type=float, default=45.034)
    parser.add_argument('--initial_azim', type=float, default=149.404)
    parser.add_argument('--camera_radius', type=float, default=142.930)
    args = parser.parse_args()

    mesh_path = choose_mesh_path(args.mesh_dir, args.frame_idx)
    mesh = trimesh.load(mesh_path, process=False)
    html = HTML_TEMPLATE
    html = html.replace('__VERTICES__', json.dumps(mesh.vertices.tolist()))
    html = html.replace('__FACES__', json.dumps(mesh.faces.tolist()))
    html = html.replace(
        '__CAMERA__',
        json.dumps(camera_from_elev_azim(
            args.initial_elev,
            args.initial_azim,
            args.camera_radius,
        )),
    )
    with open(args.output_html, 'w') as f:
        f.write(html)
    print(f'Wrote {args.output_html} using {mesh_path}')


if __name__ == '__main__':
    main()
