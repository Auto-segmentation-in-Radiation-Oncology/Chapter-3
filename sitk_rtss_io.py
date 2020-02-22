from skimage import measure
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.sequence import Sequence
import os
import numpy as np
import SimpleITK as sITK
import time
import glob
import sitk_ct_io as imio
from skimage.draw import polygon


# for debugging
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg


def read_rtss_to_sitk(rtss_file, image_dir, return_names=True, return_image=False):
    # modified code from xuefeng
    # http://aapmchallenges.cloudapp.net/forums/3/2/
    #
    # The image directory is required to set the spacing on the label map

    # read the rtss
    contours, label_names = read_contours(pydicom.read_file(rtss_file))

    # read the ct
    dcms = []
    for subdir, dirs, files in os.walk(image_dir):
        dcms = glob.glob(os.path.join(subdir, "*.dcm"))

    slices = [pydicom.read_file(dcm) for dcm in dcms]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    image = np.stack([s.pixel_array for s in slices], axis=-1)

    # convert to mask
    atlas_labels = get_mask(contours, slices, image)

    atlas_image = imio.read_sitk_image_from_dicom(image_dir)
    atlas_labels.SetOrigin(atlas_image.GetOrigin())
    atlas_labels.SetSpacing(atlas_image.GetSpacing())

    if not return_names:
        return atlas_labels
    elif not return_image:
        return atlas_labels, label_names
    else:
        return atlas_labels, label_names, atlas_image


def write_rtss_from_sitk(labels, label_names, ct_directory, output_filename):
    # labels is a sITK image volume with integer labels for the objects
    # assumes 0 for background and consequtive label numbers starting from 1
    # corresponding to the label_names
    # the ct_directory is required to correctly link the UIDs

    # load ct to get slice UIDs, z-slices and anything else we might need
    slice_info = {}
    series_info = {}
    z_values = []
    first_slice = True
    spacing = [0, 0]
    origin = [0, 0]
    with os.scandir(ct_directory) as it:
        for entry in it:
            if not entry.name.startswith('.') and entry.is_file():
                slice_file = ct_directory + entry.name
                dicom_info = pydicom.read_file(slice_file)
                slice_info[str(float(dicom_info.SliceLocation))] = dicom_info.SOPInstanceUID
                z_values.append(float(dicom_info.SliceLocation))
                if first_slice:
                    # get generic information
                    series_info['SOPClassUID'] = dicom_info.SOPClassUID
                    series_info['FrameOfReferenceUID'] = dicom_info.FrameOfReferenceUID
                    series_info['StudyInstanceUID'] = dicom_info.StudyInstanceUID
                    series_info['SeriesInstanceUID'] = dicom_info.SeriesInstanceUID
                    series_info['PatientName'] = dicom_info.PatientName
                    series_info['PatientID'] = dicom_info.PatientID
                    series_info['PatientBirthDate'] = dicom_info.PatientBirthDate
                    series_info['PatientSex'] = dicom_info.PatientSex
                    spacing[0] = float(dicom_info.PixelSpacing[0])
                    spacing[1] = float(dicom_info.PixelSpacing[1])
                    origin[0] = float(dicom_info.ImagePositionPatient[0])
                    origin[1] = float(dicom_info.ImagePositionPatient[1])
                    # Assuming axial for now
                    first_slice = False
    z_values = np.sort(z_values)

    current_time = time.localtime()
    modification_time = time.strftime("%H%M%S", current_time)
    modification_time_long = modification_time + '.123456'  # madeup
    modification_date = time.strftime("%Y%m%d", current_time)
    file_meta = Dataset()
    file_meta.FileMetaInformationGroupLength = 192
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.481.3'
    file_meta.MediaStorageSOPInstanceUID = "1.2.826.0.1.3680043.2.1125." + modification_time + ".3" + modification_date
    file_meta.ImplementationClassUID = "1.2.3.771212.061203.1"
    file_meta.TransferSyntaxUID = '1.2.840.10008.1.2'

    pydicom.dataset.validate_file_meta(file_meta, True)

    ds = FileDataset(output_filename, {},
                     file_meta=file_meta, preamble=b"\0" * 128)

    # Add the data elements
    ds.PatientName = series_info['PatientName']
    ds.PatientID = series_info['PatientID']
    ds.PatientBirthDate = series_info['PatientBirthDate']
    ds.PatientSex = series_info['PatientSex']

    # Set the transfer syntax
    ds.is_little_endian = True
    ds.is_implicit_VR = True

    # Set lots of tags
    ds.ContentDate = modification_date
    ds.SpecificCharacterSet = 'ISO_IR 100'  # probably not true TODO Check
    ds.InstanceCreationDate = modification_date
    ds.InstanceCreationTime = modification_time_long
    ds.StudyDate = modification_date
    ds.SeriesDate = modification_date
    ds.ContentTime = modification_time
    ds.StudyTime = modification_time_long
    ds.SeriesTime = modification_time_long
    ds.AccessionNumber = ''
    ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.481.3'  # RT Structure Set Stroage
    ds.SOPInstanceUID = "1.2.826.0.1.3680043.2.1125." + modification_time + ".3" + modification_date
    ds.Modality = "RTSTRUCT"
    ds.Manufacturer = "Python software"
    ds.ManufacturersModelName = 'sitk_rtss_io.py'
    ds.ReferringPhysiciansName = ''
    ds.StudyDescription = ""
    ds.SeriesDescription = "RTSS from SimpleITK data"
    ds.StudyInstanceUID = series_info['StudyInstanceUID']
    ds.SeriesInstanceUID = "1.2.826.0.1.3680043.2.1471." + modification_time + ".4" + modification_date
    ds.StructureSetLabel = "RTSTRUCT"
    ds.StructureSetName = ''
    ds.StructureSetDate = modification_time
    ds.StructureSetTime = modification_time

    contour_sequence = Sequence()
    for slice_z in z_values:
        contour_data = Dataset()
        contour_data.ReferencedSOPClassUID = series_info['SOPClassUID']
        contour_data.ReferencedSOPInstanceUID = slice_info[str(slice_z)]
        contour_sequence.append(contour_data)
    referenced_series = Dataset()
    referenced_series.SeriesInstanceUID = series_info['SeriesInstanceUID']
    referenced_series.ContourImageSequence = contour_sequence
    referenced_study = Dataset()
    referenced_study.ReferencedSOPClassUID = '1.2.840.10008.3.1.2.3.2'
    referenced_study.ReferencedSOPInstanceUID = series_info['StudyInstanceUID']
    referenced_study.RTReferencedSeriesSequence = Sequence([referenced_series])
    frame_of_ref_data = Dataset()
    frame_of_ref_data.FrameOfReferenceUID = series_info['FrameOfReferenceUID']
    frame_of_ref_data.RTReferencedStudySequence = Sequence([referenced_study])
    ds.ReferencedFrameOfReferenceSequence = Sequence([frame_of_ref_data])

    roi_sequence = Sequence()
    roi_observations = Sequence()
    for label_number in range(0, len(label_names)):
        roi_data = Dataset()
        roi_obs = Dataset()
        roi_data.ROINumber = label_number + 1
        roi_obs.ObservationNumber = label_number + 1
        roi_obs.ReferencedROINumber = label_number + 1
        roi_data.ReferencedFrameOfReferenceUID = series_info['FrameOfReferenceUID']
        roi_data.ROIName = label_names[label_number]
        roi_data.ROIObservationDescription = ''
        roi_data.ROIGenerationAlgorithm = 'Atlas-based'
        roi_data.ROIGenerationMethod = 'Python'
        roi_obs.RTROIInterpretedType = ''
        roi_obs.ROIInterpreter = ''
        roi_sequence.append(roi_data)
        roi_observations.append(roi_obs)
    ds.StructureSetROISequence = roi_sequence
    ds.RTROIObservationsSequence = roi_observations

    # as if that wasn't bad enough, now we have to add the contours!

    label_data = sITK.GetArrayFromImage(labels)

    roi_contour_sequence = Sequence()
    for label_number in range(0, len(label_names)):
        roi_contour_data = Dataset()
        roi_contour_data.ROIDisplayColor = '255\\0\\0'
        roi_contour_data.ReferencedROINumber = label_number + 1
        contour_sequence = Sequence()
        # convert labels to polygons
        contour_number = 0
        for slice_number in range(0, labels.GetSize()[2] - 1):
            slice_data = label_data[slice_number, :, :]
            slice_for_label = np.where(slice_data != label_number + 1, 0, slice_data)
            if np.any(np.isin(slice_for_label, label_number + 1)):
                contours = measure.find_contours(slice_for_label, (float(label_number + 1) / 2.0))
                for contour in contours:
                    # Convert to real world and add z_position
                    # plt.imshow(slice_data)
                    # plt.plot(contour[:, 1], contour[:, 0], color='#ff0000')
                    contour_as_string = ''
                    is_first_point = True
                    for point in contour[:-1]:
                        real_contour = [point[1] * spacing[0] + origin[0], point[0] * spacing[1] + origin[1],
                                        z_values[slice_number]]
                        if not is_first_point:
                            contour_as_string = contour_as_string + '\\'
                        else:
                            is_first_point = False
                        contour_as_string = contour_as_string + str(real_contour[0]) + '\\'
                        contour_as_string = contour_as_string + str(real_contour[1]) + '\\'
                        contour_as_string = contour_as_string + str(real_contour[2])

                    contour_number = contour_number + 1
                    contour_data = Dataset()
                    contour_data.ContourGeometricType = 'CLOSED_PLANAR'
                    contour_data.NumberOfContourPoints = str(len(contour))
                    contour_data.ContourNumber = str(contour_number)
                    image_data = Dataset()
                    image_data.ReferencedSOPClassUID = series_info['SOPClassUID']
                    image_data.ReferencedSOPInstanceUID = slice_info[str(z_values[slice_number])]
                    contour_data.ContourImageSequence = Sequence([image_data])
                    contour_data.ContourData = contour_as_string
                    contour_sequence.append(contour_data)
        roi_contour_data.ContourSequence = contour_sequence
        roi_contour_sequence.append(roi_contour_data)
    ds.ROIContourSequence = roi_contour_sequence
    ds.ApprovalStatus = 'UNAPPROVED'

    ds.save_as(output_filename)

    return


def read_contours(structure_file):
    # code from xuefeng
    # http://aapmchallenges.cloudapp.net/forums/3/2/

    contours = []
    contour_names = []
    for i in range(len(structure_file.ROIContourSequence)):
        contour = {'color': structure_file.ROIContourSequence[i].ROIDisplayColor,
                   'number': structure_file.ROIContourSequence[i].ReferencedROINumber,
                   'name': structure_file.StructureSetROISequence[i].ROIName}
        assert contour['number'] == structure_file.StructureSetROISequence[i].ROINumber
        contour['contours'] = [s.ContourData for s in structure_file.ROIContourSequence[i].ContourSequence]
        contours.append(contour)
        contour_names.append(contour['name'])

    return contours, contour_names


def get_mask(contours, slices, image):
    # code from xuefeng
    # http://aapmchallenges.cloudapp.net/forums/3/2/

    z = [s.ImagePositionPatient[2] for s in slices]

    pos_r = slices[0].ImagePositionPatient[1]
    spacing_r = slices[0].PixelSpacing[1]
    pos_c = slices[0].ImagePositionPatient[0]
    spacing_c = slices[0].PixelSpacing[0]

    im_dims = image.shape

    label = np.zeros([im_dims[2], im_dims[1], im_dims[0]], dtype=np.uint8)
    z_index = 0
    for con in contours:
        num = int(con['number'])
        for c in con['contours']:
            nodes = np.array(c).reshape((-1, 3))
            assert np.amax(np.abs(np.diff(nodes[:, 2]))) == 0
            zNew = [round(elem, 1) for elem in z]
            try:
                z_index = z.index(nodes[0, 2])
            except ValueError:
                try:
                    z_index = zNew.index(round(nodes[0, 2], 1))
                except ValueError:
                    print('Slice not found for ' + con['name'] + ' at z = ' + str(nodes[0, 2]))
            r = (nodes[:, 1] - pos_r) / spacing_r
            c = (nodes[:, 0] - pos_c) / spacing_c
            rr, cc = polygon(r, c)
            label[z_index, rr, cc] = num

    return sITK.GetImageFromArray(label)
