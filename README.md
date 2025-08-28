# PaperSummarizer

###
Login to your account on the GPU cluster, then:
```
Clone this repository: https://github.com/bipinbohara/PaperSummarizer.git
```

Step 1:
update the deployment.yaml file with your name in place of <your-name-here> and then create a deployment as follows:
```
kubectl apply -f deployment.yaml
```

Step 2:
```
kubectl exec -it paper-summary -n workspace-<your-name-here> -- /bin/bash
```

Step 3: Go to the following directory to find the file run_execution.sh
```
cd mnt/data/PaperSummary
```

Step 4: run the following script
```
chmode +x run_execution.sh

./run_execution.sh
```

#### Copy files/directory:
From local machine to GPU cluster account:
```
scp -r "/Users/username/Desktop/ResearchPapers/" sys-head-admin@131.151.54.248:~"/Desktop/ResearchPapers/"
```
Example: 
```
File will be saved from AI Agents to Papers in GPU cluster

scp -r "/Users/bbdzv/Desktop/Research Papers/AI Agents/." sys-head-admin@131.151.54.248:~"/Desktop/Papers/"
```

From GPU Machine to container:
```
kubectl cp "/home/<your-user-in-GPU-Cluster>/Desktop/Papers/." workspace-<your-name-here>/paper-summary:/mnt/data/PaperSummarizer/data -c python
```
