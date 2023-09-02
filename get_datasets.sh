echo -e "\nDownloading lm_listener_data..."

FILEID=1fR4sobslLB0gESQj6zya63XgupJFpVjc
ZIP_FILE=lm_listener_data.zip
DEST_DIR="./dataset"

EXTRACTED_DIR="extracted_contents"

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='${FILEID} -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILEID}" -O $ZIP_FILE && rm -rf /tmp/cookies.txt

unzip "$ZIP_FILE" -d "$EXTRACTED_DIR"

mkdir -p "$DEST_DIR"

mv "$EXTRACTED_DIR"/*/* "$DEST_DIR"

rm "$ZIP_FILE"
rm -r "$EXTRACTED_DIR"

echo "Downloaded and organized files into $DEST_DIR."

read -p "Press Enter to exit..."