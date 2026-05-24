import numpy as np

from magicwebpcviewer.cloud_io import load_point_cloud


def test_load_xyz_detects_rgb_and_scalar_field(tmp_path):
    path = tmp_path / "sample.xyz"
    np.savetxt(
        path,
        np.array(
            [
                [0.0, 0.0, 0.0, 255.0, 0.0, 0.0, 10.0],
                [1.0, 0.0, 0.0, 0.0, 255.0, 0.0, 20.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 255.0, 30.0],
            ]
        ),
    )

    cloud = load_point_cloud(path)

    assert cloud.point_count == 3
    assert cloud.colors is not None
    assert cloud.colors.tolist()[0] == [255, 0, 0]
    assert list(cloud.point_data) == ["field_1"]
    assert cloud.point_data["field_1"].tolist() == [10.0, 20.0, 30.0]
