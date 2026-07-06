import argparse
import os

import numpy as np


SMPLX_BODY_JOINT_NAMES = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
]

SMPLX_BODY_CONNECTIONS = [
    (0, 1), (1, 4), (4, 7), (7, 10),
    (0, 2), (2, 5), (5, 8), (8, 11),
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    (9, 13), (13, 16), (16, 18), (18, 20),
    (9, 14), (14, 17), (17, 19), (19, 21),
]

CUSTOM_JOINT_NAMES = [
    'pelvis', 'R_hip', 'R_knee', 'R_ankle',
    'L_hip', 'L_knee', 'L_ankle',
    'spine1', 'spine2', 'spine3',
    'neck', 'head', 'head_end',
    'R_shoulder_inner', 'R_shoulder', 'R_elbow', 'R_wrist',
    'L_shoulder_inner', 'L_shoulder', 'L_elbow', 'L_wrist',
]

CUSTOM_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9),
    (9, 10), (10, 11), (11, 12),
    (10, 13), (13, 14), (14, 15), (15, 16),
    (10, 17), (17, 18), (18, 19), (19, 20),
]

CUSTOM_TO_SMPLX = [
    0, 2, 5, 8, 1, 4, 7, 3, 6, 9, 12, 15, 15,
    14, 17, 19, 21, 13, 16, 18, 20,
]


def transform_skeleton(joints, axis_order, axis_signs, scale):
    axis_order = np.asarray(axis_order, dtype=np.int32)
    axis_signs = np.asarray(axis_signs, dtype=np.float32)
    return joints[..., axis_order] * axis_signs.reshape(1, 1, 3) * scale


def load_custom_skeleton_csv(path, scale, axis_order, axis_signs):
    raw_data = np.genfromtxt(path, delimiter=',', dtype=np.float32)
    if raw_data.ndim == 1:
        raw_data = raw_data.reshape(1, -1)
    raw_data = raw_data[~np.isnan(raw_data).all(axis=1)]
    if raw_data.shape[1] != 64:
        raise ValueError(f'Expected 64 CSV columns, got {raw_data.shape[1]}')
    joints = raw_data[:, 1:].reshape(-1, 21, 3)
    return transform_skeleton(joints, axis_order, axis_signs, scale)


def joint_label(index, name, extra=None):
    if extra is None:
        return f'{index}: {name}'
    return f'{index}: {name}<br>{extra}'


def line_traces(joints, connections, color):
    import plotly.graph_objects as go

    traces = []
    for start, end in connections:
        traces.append(go.Scatter3d(
            x=[joints[start, 0], joints[end, 0]],
            y=[joints[start, 1], joints[end, 1]],
            z=[joints[start, 2], joints[end, 2]],
            mode='lines',
            line=dict(width=5, color=color),
            hoverinfo='skip',
            showlegend=False,
        ))
    return traces


def joint_trace(joints, labels, color):
    import plotly.graph_objects as go

    return go.Scatter3d(
        x=joints[:, 0],
        y=joints[:, 1],
        z=joints[:, 2],
        mode='markers+text',
        marker=dict(size=5, color=color),
        text=[str(idx) for idx in range(len(joints))],
        textposition='top center',
        customdata=labels,
        hovertemplate='%{customdata}<br>x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>',
        showlegend=False,
    )


def mesh_trace(vertices, faces):
    import plotly.graph_objects as go

    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color='rgb(235, 170, 70)',
        opacity=0.35,
        flatshading=True,
        name='SMPL-X mesh',
        hoverinfo='skip',
        showlegend=False,
    )


def write_figure(traces, title, output_path):
    import plotly.graph_objects as go

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(
            aspectmode='data',
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
        ),
        width=1100,
        height=900,
        showlegend=False,
    )
    fig.write_html(output_path)
    print(f'Wrote {output_path}')


def write_smplx_debug(args):
    import torch
    import smplx

    body_model = smplx.create(
        args.model_folder,
        model_type='smplx',
        gender=args.gender,
        use_pca=False,
        flat_hand_mean=True,
    )
    with torch.no_grad():
        output = body_model(return_verts=True)

    vertices = output.vertices.detach().cpu().numpy().squeeze()
    joints = output.joints.detach().cpu().numpy().squeeze()

    joint_count = len(joints) if args.show_all_smplx_joints else min(args.smplx_joint_limit, len(joints))
    visible_joints = joints[:joint_count]
    labels = []
    for idx in range(joint_count):
        name = SMPLX_BODY_JOINT_NAMES[idx] if idx < len(SMPLX_BODY_JOINT_NAMES) else 'extra_joint'
        labels.append(joint_label(idx, name))

    connections = [
        pair for pair in SMPLX_BODY_CONNECTIONS
        if pair[0] < joint_count and pair[1] < joint_count
    ]
    traces = [
        mesh_trace(vertices, body_model.faces),
        *line_traces(visible_joints, connections, 'rgb(40, 90, 220)'),
        joint_trace(visible_joints, labels, 'rgb(220, 40, 70)'),
    ]
    write_figure(traces, f'SMPL-X Joint Numbers ({joint_count}/{len(joints)} shown)', args.smplx_output)


def write_custom_debug(args):
    skeletons = load_custom_skeleton_csv(
        args.skeleton_csv,
        args.skeleton_scale,
        args.skeleton_axis_order,
        args.skeleton_axis_signs,
    )
    if args.frame_idx < 0 or args.frame_idx >= len(skeletons):
        raise ValueError(f'frame_idx must be in [0, {len(skeletons) - 1}]')

    joints = skeletons[args.frame_idx]
    labels = [
        joint_label(idx, name, f'maps to SMPL-X {CUSTOM_TO_SMPLX[idx]}')
        for idx, name in enumerate(CUSTOM_JOINT_NAMES)
    ]
    traces = [
        *line_traces(joints, CUSTOM_CONNECTIONS, 'rgb(40, 90, 220)'),
        joint_trace(joints, labels, 'rgb(220, 40, 70)'),
    ]
    write_figure(traces, f'Custom Skeleton Joint Numbers - Frame {args.frame_idx}', args.custom_output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skeleton_csv', default='own_skeletons/_sub11_pro_processed_334.csv')
    parser.add_argument('--model_folder', default='models')
    parser.add_argument('--gender', default='neutral', choices=['neutral', 'male', 'female'])
    parser.add_argument('--frame_idx', type=int, default=0)
    parser.add_argument('--skeleton_scale', type=float, default=0.01)
    parser.add_argument('--skeleton_axis_order', nargs=3, type=int, default=[0, 1, 2])
    parser.add_argument('--skeleton_axis_signs', nargs=3, type=float, default=[-1.0, 1.0, 1.0])
    parser.add_argument('--smplx_output', default='debug_smplx_joint_numbers.html')
    parser.add_argument('--custom_output', default='debug_custom_joint_numbers.html')
    parser.add_argument('--smplx_joint_limit', type=int, default=22)
    parser.add_argument('--show_all_smplx_joints', action='store_true')
    parser.add_argument('--only', choices=['both', 'smplx', 'custom'], default='both')
    args = parser.parse_args()

    if args.only in ['both', 'smplx']:
        write_smplx_debug(args)
    if args.only in ['both', 'custom']:
        write_custom_debug(args)


if __name__ == '__main__':
    main()
