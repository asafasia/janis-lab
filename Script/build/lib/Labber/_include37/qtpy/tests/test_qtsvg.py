import pytest
from qtpy import PYSIDE6, PYQT6

def test_qtsvg():
    """Test the qtpy.QtSvg namespace"""
    from qtpy import QtSvg

    if not (PYSIDE6 or PYQT6):
        assert QtSvg.QGraphicsSvgItem is not None
        assert QtSvg.QSvgWidget is not None
    assert QtSvg.QSvgGenerator is not None
    assert QtSvg.QSvgRenderer is not None
