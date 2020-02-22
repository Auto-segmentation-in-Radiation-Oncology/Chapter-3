from __future__ import print_function
import SimpleITK as sITK
import time
import os
import numpy as np
import pydicom


def read_sitk_image_from_dicom(dir_name):
    # reading of CT based on examples from
    # https://discourse.itk.org/t/reading-dicom-series-with-simpleitk-and-filesort/1577/2
    # https://itk.org/ITKExamples/src/IO/GDCM/ReadDICOMSeriesAndWrite3DImage/Documentation.html

    series_uid = sITK.ImageSeriesReader_GetGDCMSeriesIDs(dir_name)
    uid = ''

    if len(series_uid) < 1:
        print('No DICOMs in: ' + dir_name)
        exit(1)

    if len(np.unique(series_uid)) > 1:
        print('The directory: ' + dir_name + 'contains more than one DICOM series ')
        exit(1)
    else:
        uid = np.unique(series_uid)[0]

    fileNames = sITK.ImageSeriesReader_GetGDCMSeriesFileNames(dir_name, uid)
    reader = sITK.ImageSeriesReader()
    reader.LoadPrivateTagsOn()
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.SetOutputPixelType(sITK.sitkInt32)
    reader.SetFileNames(fileNames)

    return reader.Execute()


def write_dicom_image_from_sitk(itk_image, output_directory, patient_name='ITK created', frame_of_ref=''):
    # code for writing the image modified from
    # https://simpleitk.readthedocs.io/en/next/Examples/DicomSeriesFromArray/Documentation.html

    # Write the 3D image as a series
    # IMPORTANT: There are many DICOM tags that need to be updated when you modify an
    #            original image. This is a delicate operation and requires knowledge of
    #            the DICOM standard. This example only modifies some. For a more complete
    #            list of tags that need to be modified see:
    #                           http://gdcm.sourceforge.net/wiki/index.php/Writing_DICOM
    #            If it is critical for your work to generate valid DICOM files,
    #            It is recommended to use David Clunie's Dicom3tools to validate the files
    #                           (http://www.dclunie.com/dicom3tools.html).

    writer = sITK.ImageFileWriter()
    # Use the study/series/frame of reference information given in the meta-data
    # dictionary and not the automatically generated information from the file IO
    writer.KeepOriginalImageUIDOn()

    modification_time = time.strftime("%H%M%S")
    modification_date = time.strftime("%Y%m%d")

    # Copy some of the tags and add the relevant tags indicating the change.
    # For the series instance UID (0020|000e), each of the components is a number, cannot start
    # with zero, and separated by a '.' We create a unique series ID using the date and time.
    # tags of interest:
    direction = itk_image.GetDirection()

    if frame_of_ref != '':
        # copy from image passed in
        with os.scandir(frame_of_ref) as it:
            for entry in it:
                if not entry.name.startswith('.') and entry.is_file():
                    slice1 = frame_of_ref+entry.name
                    break
        dicom_to_copy = pydicom.read_file(slice1)
        frame_of_ref_string = dicom_to_copy.FrameOfReferenceUID
        image_type = "DERIVED\\SECONDARY\\"+dicom_to_copy.ImageType[2]
    else:
        frame_of_ref_string = '1.3.6.1.4.1.14519.5.2.1.7014.4598.' + modification_date + ".1" + modification_time
        image_type = "DERIVED\\SECONDARY"

    series_tag_values = [("0008|0031", modification_time),  # Series Time
                         ("0008|0021", modification_date),  # Series Date
                         ("0008|0008", image_type),  # Image Type
                         ("0020|000d", "1.2.826.0.1.3680043.2.1125." + modification_date + ".2" + modification_time),
                         # Study Instance UID
                         ("0020|000e", "1.2.826.0.1.3680043.2.1125." + modification_date + ".1" + modification_time),
                         # Series Instance UID
                         ("0020|0037",
                          '\\'.join(map(str, (direction[0], direction[3], direction[6],  # Image Orientation (Patient)
                                              direction[1], direction[4], direction[7])))),
                         ("0020|0052", frame_of_ref_string),    # Frame of Reference UID
                         ("0008|103e", "Created-SimpleITK"),
                         ("0010|0010", patient_name),  # Patient Name
                         ("0010|0020", patient_name),  # Patient ID
                         ("0028,1054", "HU")]  # rescale type

    # Write slices to output directory
    list(map(lambda i: write_slices(series_tag_values, itk_image, i, output_directory, writer),
             range(itk_image.GetDepth())))

    return


def write_slices(series_tag_values, new_img, i, directory, writer):
    image_slice = new_img[:, :, i]

    # Tags shared by the series.
    list(map(lambda tag_value: image_slice.SetMetaData(tag_value[0], tag_value[1]), series_tag_values))

    # Slice specific tags.
    image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d"))  # Instance Creation Date
    image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S"))  # Instance Creation Time

    # Setting the type to CT preserves the slice location.
    image_slice.SetMetaData("0008|0060", "CT")  # set the type to CT so the thickness is carried over

    # (0020, 0032) image position patient determines the 3D spacing between slices.
    image_slice.SetMetaData("0020|0013", str(i))  # Instance Number
    image_slice.SetMetaData("0020|0032", '\\'.join(
        map(str, new_img.TransformIndexToPhysicalPoint((0, 0, i)))))  # Image Position (Patient)
    image_slice.SetMetaData("0020|1041", str(new_img.TransformIndexToPhysicalPoint((0, 0, i))[2]))  # slice location

    # Write to the output directory and add the extension dcm, to force writing in DICOM format.
    writer.SetFileName(os.path.join(directory, str(i) + '.dcm'))
    writer.Execute(image_slice)

    return
