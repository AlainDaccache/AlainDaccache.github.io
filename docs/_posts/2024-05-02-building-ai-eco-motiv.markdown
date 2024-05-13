---
layout: post
title:  "Building your AI Ecosystem Part Ø: Motivation"
date:   2024-05-02 09:10:02 -0400
categories: ai-ecosystem
---

## Motivation for this project

* Not many tutorials go beyond explaining the basics. Don't go beyond local development.and how they'd look like in a corporate environment.
* Little cohesiveness for an end-to-end implementation, explaining the rationale behind choices. The tooling shouldn't be an end in themselves, but only a mean, and we should understand the underlying principles since such tools change faster than the seasons.
* The double edged sword: We understand by now that most of a data scientist's time is spent on data engineering tasks, and that most models don't make it into production. 

My goal is to address these shortcomings by offering a more comprehensive approach to building an AI ecosystem, from the data architecture to the model operationalization. 

## Expectations by End of Project

By the end of this multi-part tutorial, my aim is to help you:
* Gain a deeper understanding of the key components required to establish a more data-driven infrastructure within your organization.
* Provide you with a realistic context and common use case while doing so.

In this project, we will see how an LLM can be integrated into data engineering and ML engineering tools and processes. 
We will see how to set up the right processes within your practices and get your data, people, and tools in place to maximize your chances of leveraging analytics to become more data-driven.
We will attempt at touching at the most aspects mentioned below to get an idea of the end-to-end, touching on domains from Data Engineering, Business Intelligence, Data Science, and ML Engineering.

You won't need to run anything on your side, although the GitHub repository can be found here incl. instructions to run.


## Components of the AI Ecosystem

### Data Architecture Setup

1. Data Storage: Datalake where the raw data will reside (whether unstructured, semi-structured, or structured) to ensure its fidelity as well as data warehouse for more efficient querying.
2. Data Integration: Using an orchestrator, we'll develop ELT pipelines to ingest data from fragmented sources into one place e.g. Data Lake, as well as leverage dimensional data modeling / OLAP cubes (facts & dimensions) in our data warehouse to ensure the data is well structured and consistent
3. Business Intelligence: Develop interactive reports and dashboards for analyzing the data more visually. 
4. Data Governance: We establish processes for capturing metadata to provide more context for **discovery**, **lineage**, and **quality monitoring**, and compliance (access control, data encryption & anonimization, audit trails...)

### Model Development & Deployment


1. Model Development / Experimentation: using a sandbox environment for exploratory data analysis, feature engineering, model selection and fine-tuning, as well as tooling to track model runs and performance gains over time.
2. Model Serving & Deployment: packaging and versioning the model, serving and deploying with containerization platforms or specialized cloud tooling in order to expose API endpoints or use in batch processing.
3. Workflow Orchestration: encapsulation those steps in CI/CD pipelines or other orchestrators.
4. Monitoring: Track performance metrics to detect degradation, alongside explainability tools to understand model behaviour. A/B testing possible and governance frameworks as well.


Here is a table summarizing the main area and techs we can use (bolded the ones we'll use)

| Area                   | Categpry                           | Tools                                       |
|----------------------------|---------------------------------------|---------------------------------------------|
| **Data Architecture Setup**| Data Storage                          | **MinIO**, Amazon S3, Google Cloud Storage, Azure Blob Storage, Hadoop (HDFS) |
|                            | Data Warehouse                        | Snowflake, Google BigQuery, Amazon Redshift, Azure Synapse Analytics, **Postgres**, MySQL |
|                            | Data Integration                      | **Apache Airflow**, Apache Nifi, Talend, StreamSets |
|                            | Business Intelligence                 | Power BI, Tableau, Looker, **Plotly Dash**                   |
|                            | Data Governance                      | **OpenMetadata**, Great Expectations, Apache Atlas, DataHub, Collibra, IBM InfoSphere      |
| **Model Development and Deployment**| Model Development & Experimentation| **Jupyter Notebooks**, **MLFlow**, TensorBoard        |
|                            | Model Serving & Deployment               | Flask, **FastAPI**, **Docker**          |
| **ML Orchestration**       | Workflow Orchestration                     | Jenkins, GitLab CI, GitHub Actions, **Apache Airflow**, Kubeflow, MLflow          |
| **Model Monitoring**       | Performance Monitoring                | **Prometheus**, **Grafana**, TensorBoard            |
|                            | Explainability and Interpretability   | SHAP, Lime, Alibi                           |
| **Infra & Misc.**| Container Orchestration          | Kubernetes, Docker Swarm, Amazon ECS        |
|                            | Infrastructure as Code                | Terraform, AWS CloudFormation, Azure ARM    |
|                            | Collaboration and Documentation       | Confluence, GitHub Wiki, Sphinx, **MkDocs**            |



## Areas of improvements:
- Will add support for more models for Customer Lifetime Value, Customer Churn, Customer Segmentation, and Cohort Analysis that we'll integrate in a dashboard. We'll also be developing a recommendation engine to retrieve appropriate products based on a key search, as well as recommend customers relevant products to them.
- Add in observability for ML (e.g. Prometheus) and Data (e.g. OpenMetadata)
- Cluster orchestration to allow for scalability (e.g. Kubernetes) and Infrastructure as Code for replication (e.g. Terraform)

