from __future__ import annotations

import numpy as np
import pytest
from monty.serialization import loadfn

from pymatgen.core.spectrum import Spectrum
from pymatgen.util.testing import TEST_FILES_DIR, MatSciTest


class TestIRDielectricTensor(MatSciTest):
    def setup_method(self):
        self.ir_spectra = loadfn(f"{TEST_FILES_DIR}/phonon/dos/ir_spectra_mp-991652_DDB.json")

    def test_basic(self):
        self.ir_spectra.write_json(f"{self.tmp_path}/test.json")
        ir_spectra = loadfn(f"{self.tmp_path}/test.json")
        irdict = ir_spectra.as_dict()
        ir_spectra.from_dict(irdict)

    def test_properties(self):
        assert self.ir_spectra.nph_freqs == len(self.ir_spectra.ph_freqs_gamma)
        assert self.ir_spectra.max_phfreq == max(self.ir_spectra.ph_freqs_gamma)
        assert self.ir_spectra.max_phfreq == pytest.approx(0.035711982982834185)

    def test_get_ir_spectra_default(self):
        freqs, eps = self.ir_spectra.get_ir_spectra()
        assert freqs.shape == (500,)
        assert eps.shape == (500, 3, 3)
        # imaginary part is non-trivial within the phonon-frequency range
        assert np.any(eps.imag != 0)
        # diagonal of epsilon at omega=0 is finite and includes the high-frequency contribution
        assert np.all(np.diag(eps[0].real) > 0)

    def test_get_ir_spectra_kwargs(self):
        # custom broad/emin/emax/divs
        broad_list = [1e-4] * self.ir_spectra.nph_freqs
        freqs, eps = self.ir_spectra.get_ir_spectra(broad=broad_list, emin=0.0, emax=0.05, divs=100)
        assert freqs.shape == (100,)
        assert freqs[0] == pytest.approx(0.0)
        assert freqs[-1] == pytest.approx(0.05)
        assert eps.shape == (100, 3, 3)

    def test_get_ir_spectra_bad_broad_length(self):
        with pytest.raises(ValueError, match="number of elements in the broad_list"):
            self.ir_spectra.get_ir_spectra(broad=[1e-4, 1e-4])

    @pytest.mark.parametrize(
        ("component", "reim"),
        [("xx", "re"), ("xx", "im"), ("yy", "re"), ((0, 1), "im"), ("zz", "re")],
    )
    def test_get_spectrum(self, component, reim):
        spectrum = self.ir_spectra.get_spectrum(component, reim, divs=50)
        assert isinstance(spectrum, Spectrum)
        assert spectrum.x.shape == (50,)
        assert spectrum.y.shape == (50,)

    def test_get_plotter(self):
        # `get_plotter` imports `pymatgen.vis.plotters`, which lives in the
        # umbrella `pymatgen` package (not in `pymatgen-core`). Skip when
        # running standalone.
        pytest.importorskip("pymatgen.vis.plotters")
        plotter = self.ir_spectra.get_plotter(components=("xx", "yy"), reim="reim", divs=50)
        # 2 components * 2 reim parts → 4 spectra
        assert len(plotter._spectra) == 4

    def test_get_plotter_real_only(self):
        pytest.importorskip("pymatgen.vis.plotters")
        plotter = self.ir_spectra.get_plotter(components=("xx",), reim="re", divs=50)
        assert len(plotter._spectra) == 1
        label = next(iter(plotter._spectra))
        assert label.startswith("Re")

    def test_plot(self):
        pytest.importorskip("pymatgen.vis.plotters")
        import matplotlib as mpl

        mpl.use("Agg")
        ax = self.ir_spectra.plot(components=("xx",), reim="reim", show_phonon_frequencies=True, divs=50)
        assert ax is not None
        mpl.pyplot.close("all")

    def test_plot_no_phonon_dots(self):
        pytest.importorskip("pymatgen.vis.plotters")
        import matplotlib as mpl

        mpl.use("Agg")
        ax = self.ir_spectra.plot(components=("yy",), reim="im", show_phonon_frequencies=False, divs=50)
        assert ax is not None
        mpl.pyplot.close("all")
