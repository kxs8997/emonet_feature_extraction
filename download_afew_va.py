import kagglehub

# Download latest version of the AFEW-VA dataset
path = kagglehub.dataset_download("hoanguyensgu/afew-va")

print("Path to dataset files:", path)
