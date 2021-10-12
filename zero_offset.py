r"""Create a zero offset gather for the 2D Overthrust model.

For a given number of offsets (`noffsets`) this script loads (downloads) the
2D Overthrust model and simulates the acoustic wave equation. THe source and
receiver will be placed at the same location (non-physical) to create a
zero-offset gather.

Author:
    Ali Siahkoohi (alisk@gatech.edu)
    October 2021
"""
import devito
from examples.seismic import Model, AcquisitionGeometry, TimeAxis
from examples.seismic.acoustic import AcousticWaveSolver

import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

from utils import LoadOverthrustModel

devito.configuration['log-level'] = 'WARNING'
plt.style.use('seaborn-whitegrid')
NBL = 80
SPACE_ORDER = 16
SAVE_DIRECTORY = 'computed_gather'


if __name__ == '__main__':
    # Simulation time in milliseconds.
    t0 = 0.0
    tn = 1300.0

    # Wavelet peak frequency (kHz).
    f0 = 0.015

    # Number of offsets.
    noffset = 201

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
        vp=1.0/np.sqrt(m),
        origin=origin,
        shape=shape,
        dtype=np.float32,
        spacing=spacing,
        bcs="damp",
        nbl=NBL
    )

    # Time range and number of time steps of the simulation according to CFL
    # conditions.
    time_range = TimeAxis(start=t0, stop=tn, step=model.critical_dt)
    nt = time_range.num

    # List of source/receiver locations to create zero-offset gather.
    offset_list = np.linspace(0, model.domain_size[0], num=noffset)

    # Receiver geometry.
    rec_coordinates = np.empty((1, len(model.spacing)),
                               dtype=np.float32)
    # Horizontal location of receiver.
    rec_coordinates[:, 0] = offset_list[0]
    # Vertical location of receiver.
    rec_coordinates[:, 1] = 2.0 * model.spacing[1]

    # Source geometry.
    src_coordinates = np.empty((1, len(model.spacing)),
                               dtype=np.float32)
    # Horizontal location of source.
    src_coordinates[:, 0] = offset_list[0]
    # Vertical location of source.
    src_coordinates[:, -1] = 2.0 * model.spacing[1]

    # Geometry object encoding the information on the geometry of the
    # simulation.
    geometry = AcquisitionGeometry(
        model,
        rec_coordinates,
        src_coordinates,
        t0=t0,
        tn=tn,
        src_type='Ricker',
        f0=f0
    )

    # Create finite-difference based wave-equation solver.
    solver = AcousticWaveSolver(model, geometry, space_order=SPACE_ORDER)

    # Placeholder array for the zero-offset gather.
    zero_offset_gather = np.zeros([nt, offset_list.shape[0]], dtype=np.float32)

    # Loop over the source/receiver locations to create zero-offset gather.
    with tqdm(offset_list, unit=" offsets", colour='#B5F2A9') as pb:
        for i, src_loc in enumerate(pb):

            # Assign the new source and receiver location.
            solver.geometry.src_positions[0, 0] = src_loc
            solver.geometry.rec_positions[0, 0] = src_loc

            # Compute zero-offset trace and store in the placeholder.
            zero_offset_gather[:, i] = solver.forward(
                save=False)[0].data.reshape(-1)

    # Save (overwrite) the computed zero-offset gather in a HDF5 file.
    if not os.path.exists(SAVE_DIRECTORY):
        os.makedirs(SAVE_DIRECTORY)
    hfile = h5py.File(os.path.join(SAVE_DIRECTORY, 'zero_offset_gather.h5'),
                      'w')
    hfile.create_dataset('zero_offset', data=zero_offset_gather)
    hfile.close()

    # Plotting the zero-offset gather, squared-slowness model, and the
    # perturbation model.
    extent = [
        0.0,
        model.domain_size[0] / 1.0e3,
        tn / 1.e3,
        0.
    ]
    fig = plt.figure("Zero-offset gather", dpi=100, figsize=(10, 6))
    plt.imshow(
        zero_offset_gather,
        vmin=-.6,
        vmax=.6,
        extent=extent,
        cmap="Greys",
        alpha=1.0,
        aspect=2.0,
        resample=True,
        interpolation="kaiser",
        filterrad=1
    )
    plt.xlabel("Horizontal location (km)")
    plt.ylabel("Time (s)")
    plt.title("Zero-offset gather")
    plt.grid(False)
    plt.tight_layout()

    extent = [
        0.0,
        model.domain_size[0] / 1.0e3,
        model.domain_size[1] / 1.0e3,
        0.
    ]
    fig = plt.figure("Squared-slowness model", dpi=100, figsize=(10, 5))
    plt.imshow(
        m.T,
        vmin=-0.0,
        vmax=0.2,
        extent=extent,
        cmap="YlGnBu",
        alpha=1.0,
        aspect=1.5,
        resample=True,
        interpolation="kaiser",
        filterrad=1
    )
    plt.xlabel("Horizontal location (km)")
    plt.ylabel("Depth (km)")
    plt.title(r"Squared-slowness model $(\frac{s^2}{\mathrm{km}^2})$")
    plt.colorbar(fraction=0.0218, pad=0.01)
    plt.grid(False)
    plt.tight_layout()
    plt.show()
