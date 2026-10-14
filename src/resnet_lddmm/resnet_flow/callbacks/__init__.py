"""Training callbacks for ResNetLDDMM, one subpackage per concern.

Each subpackage owns one kind of output, so what a run writes is found where
it is made, and a new callback has an obvious home:

    loss/            LossLogger, VerboseCallback       loss breakdown per step -> loss_logs/,
                                                        and printed to the console
    checkpoints/     CheckpointWriter, reading          weights -> checkpoints/, and finding /
                                                        loading them again (apply, resume)
    gradient_logs/   GradientLogger, GradientMonitor,   gradient norms, trainable parameters and
                     RequiresGradMonitor,               network layout -> grad_logs/,
                     NetworkStructureInspector          requires_grad_logs/
    deformation/     TrajectoryExporter,                the deformation itself: its path
                     DiagnosticsCallback                (trajectories/) and how well-behaved it is
                                                        (diag/* metrics)
      modes/         ModeAmplitudeLogger,               designed modes (fields/modes.py): their
                     DecompositionExporter,             amplitudes, and the trained flow split into
                     decomposition                      network and modes (mode_decomposition/),
                                                        at train end or for a finished run
    encoder/         EncoderGraphLogger, PoseLogger,    the encoder: its input graph, its predicted
                     PoseShapeExporter,                 pose as numbers and as geometry
                     EncoderPoseLogger
    schedules/       FourierAnnealing                   not a log: changes the field on a step
                                                        schedule (Fourier bands, coarse to fine)

This package imports nothing itself: import from the subpackage you need, so
using one concern (e.g. reading a checkpoint in apply) does not load the rest.
The base class and the generic callbacks shared with other trainers are in
``src.resnet_lddmm.learning.callbacks``.
"""
