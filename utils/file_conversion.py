import tempfile
from io import BytesIO
import os

def bytes_to_pdf(bytes_data: bytes) -> BytesIO:
    """
    This function is used to convert the bytes data to a pdf file.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        # Write the BytesIO content to the temporary file
        temp_file.write(bytes_data)
        temp_file_path = os.path.join(os.getcwd(), temp_file.name)
    return temp_file_path