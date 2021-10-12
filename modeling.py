r"""Simulates a shot record on the 2D Overthrust model.

This script loads (downloads) the 2D Overthrust model and simulates the
acoustic wave equation for a split-spread geometry. Source is placed in the
middle of the computational domain on slightly below the surface.

Author:
   Ali Siahkoohi (alisk@gatech.edu)
    October 2021
"""
import devito
from examples.seismic import Model, AcquisitionGeometry, TimeAxis
from examples.seismic.acoustic import AcousticWaveSolver

import matplotlib.pyplot as plt
import numpy as np

from utils import LoadOverthrustModel

devito.configuration['log-level'] = 'WARNING'
plt.style.use('seaborn-whitegrid')
NBL = 80
SPACE_ORDER = 16


if __name__ == '__main__':
    # Simulation time in milliseconds.
    t0 = 0.0
    tn = 1300.0

    # Wavelet peak frequency (kHz).
    f0 = 0.015

    # Number of source and receivers.
    nsrc = 1
    nrec = 201

    # Origin of model coordinates.
    origin = (0., 0.)

    # Load squared-slowness model.
    m = LoadOverthrustModel()

    # Grid spacing in meters.
    spacing = (12.5, 12.5)

    # Shape of the computational domain.
    shape = m.shape

    # Create Model object that encodes the information on the computational
    # domain.
    model = Model(
        space_order=SPACE_ORDER,
        vp=1.0 / np.sqrt(m),
        origin=origin,
        shape=shape,
        dtype=np.float32,
        spacing=spacing,
        bcs="damp",
        nbl=NBL
    )

    # Receiver geometry.
    rec_coordinates = np.empty((nrec, len(model.spacing)),
                               dtype=np.float32)
    # Horizontal location of receivers.
    rec_coordinates[:, 0] = np.linspace(0, model.domain_size[0],
                                        num=nrec)
    # Vertical location of receivers.
    rec_coordinates[:, 1] = 2.0 * model.spacing[1]

    # Source geometry.
    src_coordinates = np.empty((nsrc, len(model.spacing)),
                               dtype=np.float32)
    # Horizontal location of source (in the middle).
    src_coordinates[:, 0] = np.array([model.domain_size[0] // 2])
    # Vertical location of source.
    src_coordinates[:, -1] = 2.0 * model.spacing[1]

    # Geometry object encoding the information on the geometry of the
    # simulation
    geometry = AcquisitionGeometry(
        model,
        rec_coordinates,
        src_coordinates,
        t0=t0,
        tn=tn,
        src_type='Ricker',
        f0=f0
    )

    # Create finite-difference based wave-equation solver
    solver = AcousticWaveSolver(model, geometry, space_order=SPACE_ORDER)

    # Solver returns the shot record and wavefield at time t=tn (if
    # save=False) or the wavefield at all times if (save=True). Here we only
    # are interested in the shot record.
    d = solver.forward(save=False)[0].data

    # Plotting the data.
    extent = [
        -src_coordinates[:, 0][0] / 1.0e3,
        (model.domain_size[0] - src_coordinates[:, 0][0]) / 1.0e3,
        tn / 1.e3,
        0.
    ]
    fig = plt.figure("Shot record", dpi=100, figsize=(7, 10))
    plt.imshow(
        d,
        vmin=-0.6,
        vmax=0.6,
        extent=extent,
        cmap="Greys",
        alpha=1.0,
        aspect=10.0,
        resample=True,
        interpolation="kaiser",
        filterrad=1
    )
    plt.xlabel("Offset (km)")
    plt.ylabel("Time (s)")
    plt.title("Shot record")
    plt.grid(False)
    plt.tight_layout()
    plt.show()
