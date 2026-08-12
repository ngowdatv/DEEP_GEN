import os


IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".webp"
)

TEXT_EXTENSIONS = (
    ".txt",
    ".text",
    ".md",
    ".log",
    ".json"
)


def detect_dataset_type(dataset_path):

    # ========================================================
    # CHECK WHETHER PATH EXISTS
    # ========================================================

    if not os.path.exists(dataset_path):
        return "unknown"

    # ========================================================
    # IF USER PROVIDES A SINGLE FILE
    # ========================================================

    if os.path.isfile(dataset_path):

        extension = os.path.splitext(
            dataset_path
        )[1].lower()

        if extension in IMAGE_EXTENSIONS:
            return "image"

        elif extension == ".csv":
            return "csv"

        elif extension in TEXT_EXTENSIONS:
            return "text"

        else:
            return "unknown"

    # ========================================================
    # IF USER PROVIDES A DIRECTORY
    # ========================================================

    if os.path.isdir(dataset_path):

        files = []

        for root, directories, filenames in os.walk(
            dataset_path
        ):

            for filename in filenames:

                files.append(
                    filename.lower()
                )

        # ----------------------------------------------------
        # CHECK IMAGE FILES
        # ----------------------------------------------------

        for filename in files:

            if filename.endswith(
                IMAGE_EXTENSIONS
            ):

                return "image"

        # ----------------------------------------------------
        # CHECK CSV FILES
        # ----------------------------------------------------

        for filename in files:

            if filename.endswith(".csv"):

                return "csv"

        # ----------------------------------------------------
        # CHECK TEXT FILES
        # ----------------------------------------------------

        for filename in files:

            if filename.endswith(
                TEXT_EXTENSIONS
            ):

                return "text"

    return "unknown"