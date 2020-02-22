import SimpleITK as sITK
import numpy as np
import sitk_ct_io as im_io
import sitk_rtss_io as rtss_io
from os import walk, path
import os
import csv
import time


def smooth_and_resample(image, shrink_factors, smoothing_sigmas):
    """
    Args:
        image: The image we want to resample.
        shrink_factors: Number(s) greater than one, such that the new image's size is original_size/shrink_factor.
        smoothing_sigmas: Sigma(s) for Gaussian smoothing, this is in physical units, not pixels.
    Return:
        Image which is a result of smoothing the input and then resampling it using the given sigmas and shrink factors.
    """
    if np.isscalar(shrink_factors):
        shrink_factors = [shrink_factors]*image.GetDimension()
    if np.isscalar(smoothing_sigmas):
        smoothing_sigmas = [smoothing_sigmas]*image.GetDimension()

    smoothed_image = sITK.SmoothingRecursiveGaussian(image, smoothing_sigmas)

    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [int(sz/float(sf) + 0.5) for sf, sz in zip(shrink_factors, original_size)]
    new_spacing = [((original_sz-1)*original_spc)/(new_sz-1)
                   for original_sz, original_spc, new_sz in zip(original_size, original_spacing, new_size)]
    return sITK.Resample(smoothed_image, new_size, sITK.Transform(),
                         sITK.sitkLinear, image.GetOrigin(),
                         new_spacing, image.GetDirection(), 0.0,
                         image.GetPixelID())


def multiscale_demons(registration_algorithm,
                      fixed_image, moving_image, initial_transform=None,
                      shrink_factors=None, smoothing_sigmas=None, iterations=None):
    """
    Run the given registration algorithm in a multiscale fashion. The original scale should not be given as input as the
    original images are implicitly incorporated as the base of the pyramid.
    Args:
        registration_algorithm: Any registration algorithm that has an Execute(fixed_image, moving_image,
                                displacement_field_image) method.
        fixed_image: Resulting transformation maps points from this image's spatial domain to the moving image
                     spatial domain.
        moving_image: Resulting transformation maps points from the fixed_image's spatial domain to this image's
                      spatial domain.
        initial_transform: Any SimpleITK transform, used to initialize the displacement field.
        shrink_factors (list of lists or scalars): Shrink factors relative to the original image's size. When the
                                                   list entry, shrink_factors[i], is a scalar the same factor is
                                                   applied to all axes. When the list entry is a list,
                                                   shrink_factors[i][j] is applied to axis j. This allows us to
                                                   specify different shrink factors per axis. This is useful in the
                                                   context of microscopy images where it is not uncommon to have
                                                   unbalanced sampling such as a 512x512x8 image. In this case we
                                                   would only want to sample in the x,y axes and leave the z axis
                                                   as is: [[8,8,1],[4,4,1],[2,2,1]].
        smoothing_sigmas (list of lists or scalars): Amount of smoothing which is done prior to resampling the image
                                                     using the given shrink factor. These are in physical
                                                     (image spacing) units.
        iterations: a list of iterations to use at each shrink_factor. This should be the size of shrink_factors.
                    The full resolution image takes its number from the registration method passed in
    Returns:
        SimpleITK.DisplacementFieldTransform
    """

    # Create image pyramid.
    fixed_images = [fixed_image]
    moving_images = [moving_image]
    if shrink_factors:
        for shrink_factor, smoothing_sigma in reversed(list(zip(shrink_factors, smoothing_sigmas))):
            fixed_images.append(smooth_and_resample(fixed_images[0], shrink_factor, smoothing_sigma))
            moving_images.append(smooth_and_resample(moving_images[0], shrink_factor, smoothing_sigma))

    if iterations is None:
        iterations = []
        for i in range(0, len(shrink_factors)):
            iterations.append(registration_algorithm.GetNumberOfIterations())
    elif len(iterations) < len(shrink_factors):
        iterations.insert(0, registration_algorithm.GetNumberOfIterations())
        for i in range(len(iterations), len(shrink_factors)-1):
            iterations.append(iterations[-1])
    else:
        iterations.insert(0, registration_algorithm.GetNumberOfIterations())

    # Create initial displacement field at lowest resolution.
    # Currently, the pixel type is required to be sitkVectorFloat64 because of a constraint imposed by
    # the Demons filters.
    if initial_transform:
        initial_displacement_field = sITK.TransformToDisplacementField(initial_transform,
                                                                       sITK.sitkVectorFloat64,
                                                                       fixed_images[-1].GetSize(),
                                                                       fixed_images[-1].GetOrigin(),
                                                                       fixed_images[-1].GetSpacing(),
                                                                       fixed_images[-1].GetDirection())
    else:
        initial_displacement_field = sITK.Image(fixed_images[-1].GetWidth(),
                                                fixed_images[-1].GetHeight(),
                                                fixed_images[-1].GetDepth(),
                                                sITK.sitkVectorFloat64)
        initial_displacement_field.CopyInformation(fixed_images[-1])

    # Run the registration.
    registration_algorithm.SetNumberOfIterations(iterations[-1])
    displacement_field = registration_algorithm.Execute(fixed_images[-1],
                                                        moving_images[-1],
                                                        initial_displacement_field)
    # Start at the top of the pyramid and work our way down.
    for f_image, m_image, n_iterations in reversed(list(zip(fixed_images[0:-1], moving_images[0:-1],
                                                            iterations[0:-1]))):
        initial_displacement_field = sITK.Resample(displacement_field, f_image)
        registration_algorithm.SetNumberOfIterations(n_iterations)
        displacement_field = registration_algorithm.Execute(f_image, m_image, initial_displacement_field)
    return sITK.DisplacementFieldTransform(displacement_field)


def rigid_registration(fixed_image, moving_image):
    registration_method = sITK.ImageRegistrationMethod()
    registration_method.SetInitialTransform(sITK.CenteredTransformInitializer(fixed_image, moving_image,
                                                                              sITK.Similarity3DTransform()))
    registration_method.SetMetricAsCorrelation()
    registration_method.SetMetricSamplingStrategy(registration_method.REGULAR)
    registration_method.SetMetricSamplingPercentage(0.02)
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration_method.SetInterpolator(sITK.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100,
                                                      convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    return registration_method.Execute(fixed_image, moving_image)


def affine_registration(fixed_image, moving_image):
    registration_method = sITK.ImageRegistrationMethod()
    registration_method.SetInitialTransform(sITK.CenteredTransformInitializer(fixed_image, moving_image,
                                                                              sITK.AffineTransform(3)))
    registration_method.SetMetricAsCorrelation()
    registration_method.SetMetricSamplingStrategy(registration_method.REGULAR)
    registration_method.SetMetricSamplingPercentage(0.02)
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
    registration_method.SetInterpolator(sITK.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100,
                                                      convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    return registration_method.Execute(fixed_image, moving_image)


def demons_registration(fixed_image, moving_image, rigid_transform=None):
    # http: // insightsoftwareconsortium.github.io / SimpleITK - Notebooks / Python_html / 66_Registration_Demons.html
    registration_method = sITK.ImageRegistrationMethod()

    # Create initial identity transformation.
    transform_to_displacment_field_filter = sITK.TransformToDisplacementFieldFilter()
    transform_to_displacment_field_filter.SetReferenceImage(fixed_image)
    # The image returned from the initial_transform_filter is transferred to the transform and cleared out.
    initial_transform = sITK.DisplacementFieldTransform(transform_to_displacment_field_filter.Execute(sITK.Transform()))

    # Regularization (update field - viscous, total field - elastic).
    initial_transform.SetSmoothingGaussianOnUpdate(varianceForUpdateField=0.0, varianceForTotalField=2.0)

    if rigid_transform is not None:
        composite_tfm = sITK.Transform(3, sITK.sitkComposite)
        composite_tfm.AddTransform(rigid_transform)
        composite_tfm.AddTransform(initial_transform)
        registration_method.SetInitialTransform(composite_tfm)
    else:
        registration_method.SetInitialTransform(initial_transform)

    registration_method.SetMetricAsDemons(10)  # intensities are equal if the difference is less than 10HU

    # Multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[8, 4, 0])

    registration_method.SetInterpolator(sITK.sitkLinear)
    # If you have time, run this code as is, otherwise switch to the gradient descent optimizer
    # registration_method.SetOptimizerAsConjugateGradientLineSearch(learningRate=1.0, numberOfIterations=20,
    # convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=50,
                                                      convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    return registration_method.Execute(fixed_image, moving_image)


def bspline_intra_modal_registration(fixed_image, moving_image, rigid_transform=None):
    # Modified from:
    # http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/65_Registration_FFD.html

    fixed_image_mask = None

    # imageslice = imageasarray[np.uint16(fixed_image_mask.GetSize()[2]/2), :, :]
    # plt.imshow(imageslice)

    registration_method = sITK.ImageRegistrationMethod()

    # Determine the number of BSpline control points using the physical spacing we want for the control grid.
    grid_physical_spacing = [100.0, 100.0, 100.0]  # A control point every 50mm
    image_physical_size = [size * spacing for size, spacing in zip(fixed_image.GetSize(), fixed_image.GetSpacing())]
    mesh_size = [int(image_size / grid_spacing + 0.5)
                 for image_size, grid_spacing in zip(image_physical_size, grid_physical_spacing)]

    initial_transform = sITK.BSplineTransformInitializer(image1=fixed_image,
                                                         transformDomainMeshSize=mesh_size, order=3)
    if rigid_transform is not None:
        composite_tfm = sITK.Transform(3, sITK.sitkComposite)
        composite_tfm.AddTransform(rigid_transform)
        composite_tfm.AddTransform(initial_transform)
        registration_method.SetInitialTransform(composite_tfm)
    else:
        registration_method.SetInitialTransform(initial_transform)

    registration_method.SetMetricAsMeanSquares()
    # Settings for metric sampling, usage of a mask is optional. When given a mask the sample points will be
    # generated inside that region. Also, this implicitly speeds things up as the mask is smaller than the
    # whole image.
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01, seed=12312414)  # 0.05
    if fixed_image_mask:
        registration_method.SetMetricFixedMask(fixed_image_mask)

    # Multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration_method.SetInterpolator(sITK.sitkLinear)
    # registration_method.SetOptimizerAsGradientDescentLineSearch(5.0,
    #                                                            10,
    #                                                            convergenceMinimumValue=1e-4,
    #                                                            convergenceWindowSize=5)
    registration_method.SetOptimizerAsPowell(numberOfIterations=10)
    # registration_method.SetOptimizerAsLBFGS2(numberOfIterations=10)
    return registration_method.Execute(fixed_image, moving_image)


def register_images(patient_image, atlas_image, rigid=False, use_demons=False, deformable_only=False):
    # use patient as fixed and moving as atlas

    # registration following:
    # https://simpleitk.readthedocs.io/en/master/link_ImageRegistrationMethod3_docs.html

    # create initial transform
    # rigid_tfm = rigid_registration(fixed_image=sITK.Cast(patient_image, sITK.sitkFloat32),
    #                               moving_image=sITK.Cast(atlas_image, sITK.sitkFloat32))
    if not deformable_only:
        rigid_tfm = affine_registration(fixed_image=sITK.Cast(patient_image, sITK.sitkFloat32),
                                        moving_image=sITK.Cast(atlas_image, sITK.sitkFloat32))

    if not rigid:
        if use_demons:
            if deformable_only:
                # demons_filter = sITK.DemonsRegistrationFilter()
                demons_filter = sITK.DiffeomorphicDemonsRegistrationFilter()
                demons_filter.SetNumberOfIterations(10)
                # Regularization (update field - viscous, total field - elastic).
                demons_filter.SetSmoothDisplacementField(True)
                demons_filter.SetSmoothUpdateField(True)
                demons_filter.SetUpdateFieldStandardDeviations([8.0, 8.0, 1.0])
                demons_filter.SetStandardDeviations(1.0)  # [4.0, 4.0, 2.0])

                # Run the registration.
                final_tfm = multiscale_demons(registration_algorithm=demons_filter,
                                              fixed_image=sITK.Cast(patient_image, sITK.sitkFloat32),
                                              moving_image=sITK.Cast(atlas_image, sITK.sitkFloat32),
                                              smoothing_sigmas=[4, 2, 1],
                                              shrink_factors=[8, 4, 2],
                                              iterations=[100, 25, 10])

                # final_tfm = demons_registration(fixed_image=sITK.Cast(patient_image, sITK.sitkFloat32),
                #                                moving_image=sITK.Cast(atlas_image, sITK.sitkFloat32))
            else:
                final_tfm = demons_registration(fixed_image=sITK.Cast(patient_image, sITK.sitkFloat32),
                                                moving_image=sITK.Cast(atlas_image, sITK.sitkFloat32),
                                                rigid_transform=rigid_tfm)
        else:
            if deformable_only:
                final_tfm = bspline_intra_modal_registration(fixed_image=sITK.Cast(patient_image, sITK.sitkFloat32),
                                                             moving_image=sITK.Cast(atlas_image, sITK.sitkFloat32))
            else:
                final_tfm = bspline_intra_modal_registration(fixed_image=sITK.Cast(patient_image, sITK.sitkFloat32),
                                                             moving_image=sITK.Cast(atlas_image, sITK.sitkFloat32),
                                                             rigid_transform=rigid_tfm)
        if deformable_only:
            return final_tfm
        else:
            return final_tfm, rigid_tfm
    else:
        return rigid_tfm


def main():
    # Set these as desired
    write_warped_atlas = False
    write_rtss_on_warped_atlas = False
    write_rtss_on_patient_rigid = True
    write_rtss_on_patient = True
    use_demons = True
    debug = False  # this turns of the deformable so we can get to the end faster

    atlases = ['LCTSC-Train-S1-001',
               'LCTSC-Train-S1-002',
               'LCTSC-Train-S1-003',
               'LCTSC-Train-S1-004',
               'LCTSC-Train-S1-005',
               'LCTSC-Train-S1-006',
               'LCTSC-Train-S1-007',
               'LCTSC-Train-S1-008',
               'LCTSC-Train-S1-009',
               'LCTSC-Train-S1-010',
               'LCTSC-Train-S1-011',
               'LCTSC-Train-S1-012',
               'LCTSC-Train-S2-001',
               'LCTSC-Train-S2-002',
               'LCTSC-Train-S2-003',
               'LCTSC-Train-S2-004',
               'LCTSC-Train-S2-005',
               'LCTSC-Train-S2-006',
               'LCTSC-Train-S2-007',
               'LCTSC-Train-S2-008',
               'LCTSC-Train-S2-009',
               'LCTSC-Train-S2-010',
               'LCTSC-Train-S2-011',
               'LCTSC-Train-S2-012',
               'LCTSC-Train-S3-001',
               'LCTSC-Train-S3-002',
               'LCTSC-Train-S3-003',
               'LCTSC-Train-S3-004',
               'LCTSC-Train-S3-005',
               'LCTSC-Train-S3-006',
               'LCTSC-Train-S3-007',
               'LCTSC-Train-S3-008',
               'LCTSC-Train-S3-009',
               'LCTSC-Train-S3-010',
               'LCTSC-Train-S3-011',
               'LCTSC-Train-S3-012',
               'LCTSC-Test-S1-101',
               'LCTSC-Test-S1-102',
               'LCTSC-Test-S1-103',
               'LCTSC-Test-S1-104',
               'LCTSC-Test-S2-101',
               'LCTSC-Test-S2-102',
               'LCTSC-Test-S2-103',
               'LCTSC-Test-S2-104',
               'LCTSC-Test-S3-101',
               'LCTSC-Test-S3-102',
               'LCTSC-Test-S3-103',
               'LCTSC-Test-S3-104']

    patients = ['LCTSC-Test-S1-201',
                'LCTSC-Test-S1-202',
                'LCTSC-Test-S1-203',
                'LCTSC-Test-S1-204',
                'LCTSC-Test-S2-201',
                'LCTSC-Test-S2-202',
                'LCTSC-Test-S2-203',
                'LCTSC-Test-S2-204',
                'LCTSC-Test-S3-201',
                'LCTSC-Test-S3-202',
                'LCTSC-Test-S3-203',
                'LCTSC-Test-S3-204']

    root_dir = 'C:\\Mark\\Book\\TidyChallengeData\\'
    # output_root_dir = 'C:\\Mark\\Book\\Atlas selection code\\ResultsDataBSpline\\'
    output_root_dir = 'C:\\Mark\\Book\\Atlas selection code\\ResultsDataDemons\\'

    test_patient_ct = ''
    test_patient_rtss = ''
    test_atlas_ct = ''
    test_atlas_rtss = ''

    for patient in patients:
        print('Processing case: ' + patient)
        print('')

        test_patient_path = root_dir + patient + '\\'
        output_patient_dir = output_root_dir + patient + '\\'
        if not path.exists(output_patient_dir):
            os.mkdir(output_patient_dir)
        for (dirpath, dirnames, filenames) in walk(test_patient_path):
            if len(filenames) > 1:
                test_patient_ct = dirpath + '\\'
            if len(filenames) == 1:
                test_patient_rtss = dirpath + '\\' + filenames[0]

        for atlas in atlases:
            print('   Processing atlas: ' + atlas)
            start_time = time.time()

            test_atlas_path = root_dir + atlas + '\\'
            if not path.exists(output_root_dir + patient + '\\' + atlas + '\\'):
                os.mkdir(output_root_dir + patient + '\\' + atlas + '\\')
            output_csv = output_root_dir + patient + '\\' + atlas + '\\results.csv'
            if os.path.exists(output_csv):
                # skip if we've already done this atlas patient combo
                # useful if we have to stop processing halfway
                print('Result found. Skipped')
                print('')
                continue
            output_dir = output_root_dir + patient + '\\' + atlas + '\\CT\\'
            if not path.exists(output_dir):
                os.mkdir(output_dir)
            if not path.exists(output_root_dir + patient + '\\' + atlas + '\\RTSS\\'):
                os.mkdir(output_root_dir + patient + '\\' + atlas + '\\RTSS\\')
            if not path.exists(output_root_dir + patient + '\\' + atlas + '\\RTSS_rigid\\'):
                os.mkdir(output_root_dir + patient + '\\' + atlas + '\\RTSS_rigid\\')
            output_rtss = output_root_dir + patient + '\\' + atlas + '\\RTSS\\IM1.dcm'
            output_rigid_rtss = output_root_dir + patient + '\\' + atlas + '\\RTSS_rigid\\IM1.dcm'
            for (dirpath, dirnames, filenames) in walk(test_atlas_path):
                if len(filenames) > 1:
                    test_atlas_ct = dirpath + '\\'
                if len(filenames) == 1:
                    test_atlas_rtss = dirpath + '\\' + filenames[0]

            # load the current atlas and structures
            # print('loading atlas')
            atlas_labels, atlas_label_names, atlas_image = rtss_io.read_rtss_to_sitk(test_atlas_rtss, test_atlas_ct,
                                                                                     True, True)

            # put underscores in spaces for names
            # deals with a naming error in the original data
            atlas_label_names = [name.replace(' ', '_') for name in atlas_label_names]

            # load the current patient
            # print('loading patient')
            patient_labels, patient_label_names, patient_image = rtss_io.read_rtss_to_sitk(test_patient_rtss,
                                                                                           test_patient_ct,
                                                                                           True, True)
            # put underscores in spaces for names
            # deals with a naming error in the original data
            patient_label_names = [name.replace(' ', '_') for name in patient_label_names]

            resampler = sITK.ResampleImageFilter()

            # Rigidly register the images
            # Doing a composite registration of the rigid+deformable is technically more correct, but appears
            # to take a lot longer in ITK. Resampling the atlas after rigid and using those images/labels is much faster
            # The registration function needs modifying to make this cleaner

            rigid_tfm = register_images(patient_image, atlas_image, True)

            # resampler.SetTransform(rigid_tfm)
            # resampler.SetReferenceImage(patient_image)
            # resampler.SetInterpolator(sITK.sitkNearestNeighbor)
            # atlas_labels_rigid = resampler.Execute(atlas_labels)

            # rigidly resample the atlas image to the patient image
            resampler.SetTransform(rigid_tfm)
            resampler.SetReferenceImage(patient_image)
            resampler.SetDefaultPixelValue(-1000)
            resampler.SetInterpolator(sITK.sitkLinear)
            atlas_image_rigid = resampler.Execute(atlas_image)

            # deformably register the images
            # registration_tfm, rigid_tfm = register_images(patient_image, atlas_image_rigid, debug, use_demons=True,
            #                                              deformable_only=True)
            registration_tfm = register_images(patient_image, atlas_image_rigid, debug, use_demons=use_demons,
                                               deformable_only=True)

            # rigidly resample the atlas image to the patient image
            resampler.SetTransform(rigid_tfm)
            resampler.SetReferenceImage(patient_image)
            resampler.SetDefaultPixelValue(-1000)
            resampler.SetInterpolator(sITK.sitkLinear)
            atlas_image_rigid = resampler.Execute(atlas_image)

            # warp the atlas image to the patient image
            resampler.SetTransform(registration_tfm)
            resampler.SetReferenceImage(patient_image)
            resampler.SetDefaultPixelValue(-1000)
            resampler.SetInterpolator(sITK.sitkLinear)
            atlas_image_deform = resampler.Execute(atlas_image_rigid)

            if write_warped_atlas:
                im_io.write_dicom_image_from_sitk(atlas_image_deform, output_dir, 'Warped Image', test_patient_ct)

            resampler.SetTransform(rigid_tfm)
            resampler.SetReferenceImage(patient_image)
            resampler.SetInterpolator(sITK.sitkNearestNeighbor)
            atlas_labels_rigid = resampler.Execute(atlas_labels)

            # transfer labels to the patient image
            resampler.SetTransform(registration_tfm)
            resampler.SetReferenceImage(patient_image)
            resampler.SetInterpolator(sITK.sitkNearestNeighbor)
            atlas_labels_deform = resampler.Execute(atlas_labels_rigid)

            if write_rtss_on_patient_rigid:
                rtss_io.write_rtss_from_sitk(atlas_labels_rigid, atlas_label_names, test_patient_ct, output_rigid_rtss)
            if write_rtss_on_patient:
                rtss_io.write_rtss_from_sitk(atlas_labels_deform, atlas_label_names, test_patient_ct, output_rtss)
            if write_rtss_on_warped_atlas & write_warped_atlas:
                rtss_io.write_rtss_from_sitk(atlas_labels_deform, atlas_label_names, output_dir, output_rtss)

            # calculate similarity measures
            dsc = {}
            nmi_before_dir = {}
            rmse_before_dir = {}
            nmi_after_dir = {}
            rmse_after_dir = {}
            nmi_local = {}
            rmse_local = {}
            for atlas_label_number in range(1, len(atlas_label_names) + 1):
                label_name = atlas_label_names[atlas_label_number - 1]
                # patient structures may not be the same order as the atlas ones
                patient_label_number = next((patient_label_number for patient_label_number in
                                             range(1, len(patient_label_names) + 1) if
                                             patient_label_names[patient_label_number - 1] == label_name), -1)
                if patient_label_number == -1:
                    # the structure name wasn't found
                    dsc[label_name] = -1
                    nmi_before_dir[label_name] = -1
                    rmse_before_dir[label_name] = -1
                    nmi_after_dir[label_name] = -1
                    rmse_after_dir[label_name] = -1
                else:
                    # calculate dsc
                    label_arr_1 = sITK.GetArrayFromImage(atlas_labels_deform)
                    label_arr_2 = sITK.GetArrayFromImage(patient_labels)
                    dsc[label_name] = 2 * np.count_nonzero((label_arr_1 == atlas_label_number) &
                                                           (label_arr_2 == patient_label_number)) / \
                                      (np.count_nonzero(label_arr_1 == atlas_label_number) +
                                       np.count_nonzero(label_arr_2 == patient_label_number))
                    # calculate image similarity measures (global)
                    image_arr_1 = sITK.GetArrayFromImage(atlas_image_rigid)
                    image_arr_2 = sITK.GetArrayFromImage(atlas_image_deform)
                    image_arr_3 = sITK.GetArrayFromImage(patient_image)
                    # clear the out of bounds to air
                    image_arr_1 = np.where(image_arr_1 < -1000, -1000, image_arr_1)
                    image_arr_2 = np.where(image_arr_2 < -1000, -1000, image_arr_2)
                    image_arr_3 = np.where(image_arr_3 < -1000, -1000, image_arr_3)
                    # average ssd over the whole image (averaged so that it is less impacted by image size)
                    rmse_before_dir[label_name] = np.sqrt(np.sum(np.square(image_arr_1 - image_arr_3) /
                                                                 np.size(image_arr_1)))
                    rmse_after_dir[label_name] = np.sqrt(np.sum(np.square(image_arr_2 - image_arr_3) /
                                                                np.size(image_arr_2)))
                    rmse_local[label_name] = np.sqrt(np.sum(np.square(image_arr_2 - image_arr_3) /
                                                            np.count_nonzero([label_arr_1 == atlas_label_number]),
                                                            where=[label_arr_1 == atlas_label_number]))

                    # more borrowed code
                    # https://matthew-brett.github.io/teaching/mutual_information.html
                    hist_imgs_1_3, xedges, yedges = np.histogram2d(image_arr_1.ravel(), image_arr_3.ravel(),
                                                                   32, density=False)
                    pxy = hist_imgs_1_3
                    pxy = pxy / np.size(image_arr_1)
                    px = np.sum(pxy, axis=1)  # marginal for x over y
                    py = np.sum(pxy, axis=0)  # marginal for y over x
                    px_py = px[:, None] * py[None, :]  # Broadcast to multiply marginals
                    # Now we can do the calculation using the pxy, px_py 2D arrays
                    nzs = pxy > 0  # Only non-zero pxy values contribute to the sum
                    hxy = -np.sum(pxy[nzs] * np.log(pxy[nzs]))
                    ixy = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
                    nmi_before_dir[label_name] = (ixy / hxy)

                    hist_imgs_2_3, xedges, yedges = np.histogram2d(image_arr_2.ravel(), image_arr_3.ravel(),
                                                                   32, density=True)
                    pxy = hist_imgs_2_3
                    pxy = pxy / np.size(image_arr_1)
                    px = np.sum(pxy, axis=1)  # marginal for x over y
                    py = np.sum(pxy, axis=0)  # marginal for y over x
                    px_py = px[:, None] * py[None, :]  # Broadcast to multiply marginals
                    # Now we can do the calculation using the pxy, px_py 2D arrays
                    nzs = pxy > 0  # Only non-zero pxy values contribute to the sum
                    hxy = -np.sum(pxy[nzs] * np.log(pxy[nzs]))
                    ixy = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
                    nmi_after_dir[label_name] = (ixy / hxy)

                    # local NMI is more difficult as we have to build the right histograms
                    # note we can't use the patient labels as we wouldn't know those
                    image_ass_2l = image_arr_2.ravel()[label_arr_1.ravel() == atlas_label_number]
                    image_ass_3l = image_arr_3.ravel()[label_arr_1.ravel() == atlas_label_number]
                    hist_imgs_2_3_l, xedges, yedges = np.histogram2d(image_ass_2l, image_ass_3l,
                                                                     32, density=True)
                    pxy = hist_imgs_2_3_l
                    pxy = pxy / np.size(image_ass_2l)
                    px = np.sum(pxy, axis=1)  # marginal for x over y
                    py = np.sum(pxy, axis=0)  # marginal for y over x
                    px_py = px[:, None] * py[None, :]  # Broadcast to multiply marginals
                    # Now we can do the calculation using the pxy, px_py 2D arrays
                    nzs = pxy > 0  # Only non-zero pxy values contribute to the sum
                    hxy = -np.sum(pxy[nzs] * np.log(pxy[nzs]))
                    ixy = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
                    nmi_local[label_name] = (ixy / hxy)

            # save the results
            print('Writing results to: ', output_csv)
            with open(output_csv, mode='w', newline='\n', encoding='utf-8') as out_file:
                result_writer = csv.writer(out_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                result_writer.writerow(['Organ', 'DSC', 'NMI Rigid', 'RMSE Rigid', 'NMI Deform', 'RMSE Deform',
                                        'NMI ROI', 'RMSE ROI'])
                for label_name in atlas_label_names:
                    result_writer.writerow([label_name,
                                            dsc[label_name],
                                            nmi_before_dir[label_name],
                                            rmse_before_dir[label_name],
                                            nmi_after_dir[label_name],
                                            rmse_after_dir[label_name],
                                            nmi_local[label_name],
                                            rmse_local[label_name]])

            elapsed_time = time.time() - start_time
            print('   Time taken: {:.2f} mins'.format(elapsed_time / 60))
            print('')

    return


if __name__ == '__main__':
    main()
