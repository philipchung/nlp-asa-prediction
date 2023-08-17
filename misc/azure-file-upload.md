# Copying Files to Azure Storage

## AzCopy
AzCopy is a command line utility for copying files to and from Azure Storage.  This is useful for headless or command-line only environments.

```sh
#Download AzCopy
wget https://aka.ms/downloadazcopy-v10-linux
 
#Expand Archive
tar -xvf downloadazcopy-v10-linux
 
#(Optional) Remove existing AzCopy version
sudo rm /usr/bin/azcopy
 
#Move AzCopy to the destination you want to store it
sudo cp ./azcopy_linux_amd64_*/azcopy /usr/bin/
```

Now log-in and provide authentication to storage account.
To do this, go to [portal.azure.com](portal.azure.com), identify the blob store in the storage account you want to upload data to.  Then navigate to "Shared access tokens" and create a new shared access token.  Make sure "Write" is enabled under permissions.  These tokens are valid for a limited time range and may be restricted to certain IP addresses.

```sh
# Authenticate with Azure AD
azcopy cp "/local/path" "<replace with blobstore SAS URL>" --recursive=true

# Example with fake SAS token
azcopy cp "/workspace/data/id/v3/raw" "https://stamlbuild2742hf.blob.core.windows.net/azureml-blobstore-5bb6c513-1753-4788-911a-616beb4fab1f?sp=racwdli&st=2022-04-28T03:34:47Z&se=2022-05-01T11:34:47Z&spr=https&sv=2020-08-04&sr=c&sig=oH7LnFtLUWhX219YPe4FKmvf8Pk1HzYzU44VvH10nJk%3D" --recursive=true

```

## Azure Storage Explorer
[Azure storage explorer](https://azure.microsoft.com/en-us/features/storage-explorer/#overview) is a desktop app that can be downloaded and data can be uploaded via drag-and-drop UI.  Furthermore, it has powerful capabilities for deleting blobs, creating virtual folders, etc.

## References:
* https://azure.microsoft.com/en-us/blog/azcopy-on-linux-now-generally-availble/
* https://www.thomasmaurer.ch/2019/05/how-to-install-azcopy-for-azure-storage/