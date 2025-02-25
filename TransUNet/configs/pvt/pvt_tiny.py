cfg = dict(
    model='pvt_tiny',
    drop_path=0.1,
    clip_grad=None,
    output_dir='checkpoints/pvt_tiny',
    backbone=dict(
        type='pvt_tiny',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='../../pretrained/pvt_tiny.pth'
        )
    )
)