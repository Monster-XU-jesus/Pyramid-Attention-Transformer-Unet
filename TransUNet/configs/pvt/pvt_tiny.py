cfg = dict(
    model='pvt_tiny',
    drop_path=0.1,
    clip_grad=None,
    output_dir='checkpoints/pvt_tiny',
    load_from='../../pretrained/pvt_tiny.pth',  # 新增加载路径配置
    backbone=dict(
        type='pvt_tiny',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='../../pretrained/pvt_tiny.pth'
        )
    )
)