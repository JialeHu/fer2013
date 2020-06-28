rm /home/hjljjdd_gmail_com/log.txt

python /home/hjljjdd_gmail_com/ImageLearning.py >> log.txt

gsutil cp /home/hjljjdd_gmail_com/log.txt gs://fer2013-data
gsutil cp /home/hjljjdd_gmail_com/ImageNet4.py gs://fer2013-data
gsutil cp /home/hjljjdd_gmail_com/ImageNet4.pt gs://fer2013-data

gcloud compute instances stop pytorch-cpu-vm --zone=us-east4-b
