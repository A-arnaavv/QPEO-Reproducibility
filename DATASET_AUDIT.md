# Dataset audit for the reconstructed experiments

This document records what the official public sources actually provide. It is kept
separate from the old manuscript claims so the rebuilt experiments do not silently
inherit unsupported dataset descriptions.

## Google Cluster Data v3
Official source: https://github.com/google/cluster-data

The v3 schema provides task/instance events, CPU and memory requests/usages, machine
placement and related metadata. It does **not** expose a native GPU-request field in
the published v3 resource structure. Therefore a GPU demand used by QPEO must be
an explicitly documented reconstruction mapping rather than described as directly
observed GPU demand from the trace.

## Alibaba GPU trace v2023
Official source: https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2023

This is a directly relevant GPU scheduling trace. The released pod CSV includes CPU
request, memory request, number of GPUs, fractional GPU request (`gpu_milli`), QoS,
and creation/scheduling/deletion times. The node CSV includes CPU, memory, GPU count
and GPU model. This should replace the ambiguous manuscript label "Alibaba Cluster
Trace 2022" for the GPU-specific experiment.

## Microsoft Azure
Official source: https://github.com/Azure/AzurePublicDataset

`AzurePublicDatasetV2` is a 2019 **VM trace**, not a serverless invocation trace. For
a serverless experiment, use an Azure Functions trace. The reconstruction currently
targets `AzureFunctionsInvocationTrace2021`, which provides invocation completion
time and duration. It does not expose CPU/GPU/memory/bandwidth demand, so any such
resource demand used in QPEO must be supplied by an explicit reconstruction policy.

## Kubernetes HPA
The Kubernetes `perf-tests` and autoscaler repositories provide benchmarking tools
and HPA implementations, but the dataset audit did not identify an official public
production trace matching the old manuscript claim of "Kubernetes HPA Traces" with
~320 nodes and ~412K pods. The rebuilt study should therefore either:

1. label a generated HPA workload as a **synthetic/replay benchmark**, with its
   generator configuration published; or
2. replace this benchmark with a verifiable public trace.

It should not be described as a public production trace unless a citable source is
identified.
