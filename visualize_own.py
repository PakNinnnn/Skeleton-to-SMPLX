import argparse
import os

import numpy as np
import plotly.graph_objects as go
import trimesh


CONNECTIONS = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9),
    (9, 10), (10, 11), (11, 12),
    (10, 13), (13, 14), (14, 15), (15, 16),
    (10, 17), (17, 18), (18, 19), (19, 20),
]

JOINT_LABELS = [
    'pelvis', 'R_hip', 'R_knee', 'R_ankle',
    'L_hip', 'L_knee', 'L_ankle',
    'spine1', 'spine2', 'spine3',
    'neck', 'head', 'head_end',
    'R_shoulder_inner', 'R_shoulder', 'R_elbow', 'R_wrist',
    'L_shoulder_inner', 'L_shoulder', 'L_elbow', 'L_wrist',
]


def load_skeleton_csv(path, scale):
    raw_data = np.genfromtxt(path, delimiter=',', dtype=np.float32)
    if raw_data.ndim == 1:
        raw_data = raw_data.reshape(1, -1)
    raw_data = raw_data[~np.isnan(raw_data).all(axis=1)]
    if raw_data.shape[1] != 64:
        raise ValueError(f'Expected 64 CSV columns, got {raw_data.shape[1]}')
    return raw_data[:, 1:].reshape(-1, 21, 3) * scale


def build_skeleton(joints):
    traces = [
        go.Scatter3d(
            x=[joints[start, 0], joints[end, 0]],
            y=[joints[start, 1], joints[end, 1]],
            z=[joints[start, 2], joints[end, 2]],
            name=JOINT_LABELS[start],
            mode='lines',
            line=dict(width=5, color='rgb(50, 120, 220)'),
            opacity=0.85,
            showlegend=False,
        )
        for start, end in CONNECTIONS
    ]
    traces.append(go.Scatter3d(
        x=joints[:, 0],
        y=joints[:, 1],
        z=joints[:, 2],
        mode='markers',
        marker=dict(size=4, color='rgb(220, 50, 80)'),
        text=JOINT_LABELS,
        hoverinfo='text',
        showlegend=False,
    ))
    return traces


def build_mesh(mesh):
    return go.Mesh3d(
        x=mesh.vertices[:, 0],
        y=mesh.vertices[:, 1],
        z=mesh.vertices[:, 2],
        i=mesh.faces[:, 0],
        j=mesh.faces[:, 1],
        k=mesh.faces[:, 2],
        color='rgb(245, 170, 60)',
        opacity=0.45,
        flatshading=True,
        showlegend=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skeleton_csv', required=True)
    parser.add_argument('--mesh_dir', required=True)
    parser.add_argument('--output_html', default='own_skeleton_smplx.html')
    parser.add_argument('--skeleton_scale', type=float, default=0.01)
    args = parser.parse_args()

    skeletons = load_skeleton_csv(args.skeleton_csv, args.skeleton_scale)
    mesh_paths = sorted(
        os.path.join(args.mesh_dir, name)
        for name in os.listdir(args.mesh_dir)
        if name.lower().endswith('.obj')
    )
    meshes = [trimesh.load(path, process=False) for path in mesh_paths]
    num_frames = min(len(skeletons), len(meshes))
    if num_frames == 0:
        raise ValueError('No overlapping skeleton frames and meshes to visualize')

    frames = []
    for idx in range(num_frames):
        frames.append(go.Frame(
            data=[build_mesh(meshes[idx])] + build_skeleton(skeletons[idx]),
            name=str(idx),
        ))

    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=go.Layout(
            title='Custom Skeleton and SMPL-X Mesh',
            scene=dict(aspectmode='data'),
            width=1100,
            height=900,
            updatemenus=[dict(
                type='buttons',
                buttons=[
                    dict(
                        label='Play',
                        method='animate',
                        args=[None, {'frame': {'duration': 120, 'redraw': True}}],
                    ),
                    dict(
                        label='Pause',
                        method='animate',
                        args=[[None], {'frame': {'duration': 0, 'redraw': False},
                                       'mode': 'immediate'}],
                    ),
                ],
            )],
            sliders=[dict(
                steps=[
                    dict(
                        method='animate',
                        args=[[str(idx)], {'mode': 'immediate',
                                           'frame': {'duration': 0, 'redraw': True}}],
                        label=str(idx),
                    )
                    for idx in range(num_frames)
                ],
            )],
        ),
    )
    fig.write_html(args.output_html)
    print(f'Wrote {args.output_html}')


if __name__ == '__main__':
    main()
