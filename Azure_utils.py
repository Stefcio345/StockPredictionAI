from azure.storage.blob import BlobServiceClient


def download_model():
    connection_string = ""
    # Set these values
    container_name = "container"
    blob_name = "trained_multi_target_models.pkl"
    download_file_path = "downloaded_model.pkl"

    # Create client
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)

    # Download blob
    with open(download_file_path, "wb") as file:
        data = container_client.download_blob(blob_name)
        file.write(data.readall())

    print(f"Downloaded '{blob_name}' to '{download_file_path}'")