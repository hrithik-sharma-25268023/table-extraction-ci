"""Testing Module"""

import os
PDF_PATH = "/home/hrithik-dev/ds-labs/table-extraction-ci/data/images/American-Express-Annual-Report-2025.pdf"


def test_images():
    """test number of images"""

    assert len(os.listdir(PDF_PATH)) == 192
