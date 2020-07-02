sudo docker build -t docker_porous_test .
sudo docker run docker_porous_test

## list images and containers
#sudo docker image ls # list image ID
#sudo docker ps -a # list container ID

## remove images and stuck containers
#sudo docker rmi "image_id"
#sudo docker rm "container_id"

## remove all images and containers
#sudo docker rm $(sudo docker ps -a -q)
#sudo docker rmi $(sudo docker image ls -q)

## copy docker file --> based on container ID
#sudo docker cp bfba3ce73227:/VP_results VP_results

## provide acces to files copied from docker container
#sudo chown -R $username VP_results
