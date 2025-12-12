import SimpleITK as sitk


def convert_LPI_to_RAI(image):

    arr = sitk.GetArrayFromImage(image)

    spacing = image.GetSpacing()
    origin = image.GetOrigin()
    direction = image.GetDirection()

    print("Original spacing:", spacing)
    print("Original direction:", direction)
    print("Original origin:", origin)

    arr_hfs = arr[:, ::-1, ::-1]

    image_hfs = sitk.GetImageFromArray(arr_hfs)
    image_hfs.SetSpacing(spacing)
    image_hfs.SetOrigin(origin)
    image_hfs.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))

    return image_hfs


if __name__ == "__main__":
    input_file = r'D:\Data\Test\case9\ct.nii.gz'
    image = sitk.ReadImage(input_file)

    image_hfs = convert_LPI_to_RAI(image)

    # 如果需要保存
    output_file = input_file.replace('.nii.gz', '_2.nii.gz')
    sitk.WriteImage(image_hfs, output_file)
    print("Saved HFS mask to:", output_file)
