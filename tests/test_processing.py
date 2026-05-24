import numpy as np

from magicwebpcviewer.models import CloudData
from magicwebpcviewer.processing import available_scalar_fields, compute_properties, voxel_downsample


def test_voxel_downsample_keeps_one_point_per_voxel():
    cloud = CloudData(
        name="sample",
        source_format="xyz",
        points=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.1, 0.1],
                [1.0, 1.0, 1.0],
            ]
        ),
        point_data={"intensity": np.array([1, 2, 3])},
    )

    result = voxel_downsample(cloud, 0.5)

    assert result.original_count == 3
    assert result.downsampled_count == 2
    assert result.cloud.points.shape == (2, 3)
    assert result.cloud.point_data["intensity"].tolist() == [1, 3]


def test_compute_properties_reports_scalar_fields_and_bounds():
    cloud = CloudData(
        name="sample",
        source_format="xyz",
        points=np.array([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]]),
        point_data={"classification": np.array([2, 5])},
    )

    props = compute_properties(cloud)

    assert available_scalar_fields(cloud) == ["classification"]
    assert props["point_count"] == 2
    assert np.allclose(props["bounds_min"], [0.0, 1.0, 2.0])
    assert np.allclose(props["bounds_max"], [2.0, 3.0, 4.0])
