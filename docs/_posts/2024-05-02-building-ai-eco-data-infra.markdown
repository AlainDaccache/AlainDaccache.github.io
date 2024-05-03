---
layout: post
title:  "Building your AI Ecosystem Part I: Data Architecture"
date:   2024-05-02 11:10:02 -0400
categories: ai-ecosystem data-architecture mongodb postgres airflow minio
---



Suppose we're a retail company. We have some internal data that employees make use of when understanding internal procedures and  guidelines. They're in the form of pdfs within a fileserver, such as FTP and SFTP. r data scientists that want to build an LLM. Our data scientists also want access to customer transactions data in order to build models that would help forecast their lifetime value and churn risk in order to prioritize the marketing team's targeting initiatives.

While the DSs can gain access to the FTP and MongoDB servers, this adds an extra layers of security risk; the IT team is better off giving access to a centralized area to increase control while helping the DSs navigate the company's data more accessibly.

To simplify the FTP server, we created a simple Python Flask Server that will serve our files. We'll initially populate it by using the script under `scripts/online_data_to_ftp_and_mongodb.py`, which, like the name suggest, we also populate our MongoDB database; here's a sample of the HR procedure:

![HR Guideline Sample](/assets/hr-guideline-sample.png)

And a sample of the MongoDB database's data:

```
{
    _id: ObjectId('6634ed99e4cf3b4b5d6076d5'),
    InvoiceNo: '536365',
    StockCode: '85123A',
    Description: 'WHITE HANGING HEART T-LIGHT HOLDER',
    Quantity: 6,
    InvoiceDate: '12/1/2010 8:26',
    UnitPrice: 2.55,
    CustomerID: 17850,
    Country: 'United Kingdom'
}
```

You can find the instructions on how to setup the project in the `README.md` file of the [project](https://github.com/AlainDaccache/CustomerIsKing).

After we populate our "data sources", our next step is to integrate it in a centralized repository. While possible to simply write a script (e.g. in Bash or Python) that will have some authentication to connect to those sources and then dump it in a data warehouse, remember, our data isn't structure per-se.
1. The unstructured data (`.pdf`) doesn't need to be analyzed and dumped in some tabular format right away. We want to preserve all the informationso then the data scientists can parse it and use it as part of an LLM.
2. The semi-structured data (`.json`-like) could be converted to tabular form, but in the case of nested data, isn't straightforward to dump into a data warehouse right away (although I've seen it, since some DWH like nowflake support more complex data types to accomodate for that).

Either way, an intermediary that is flexible enough to store the data in a raw form and different formats, could be useful in that situation. This is the data lake, and we have multiple technologies available, such as **AWS S3**, **GCP Object Storage**, **Azure Blob Storage**, and **MinIO** for a cloud-native solution. We will be using the latter, but please note it has a *GNU AGPL v3* license, which means it is to be used for open-source only if not paying for a commercial license. Don't sweat, it is AWS-S3 compatible (shares the same API as AWS) so it will be easier to reuse in your context, I've seen it used at a govenment agency.

For this integration i.e. from FTP and MongoDB to the data lake (MinIO), we will be using a workflow orchestrator to will be able to manage the different pipelines you'd be using, whether it's for data integration, ML training etc. We're going with **Apache Airflow** due to its flexibility compared to more opaque or low-code/no-code solutions. Here's a sample of such code (that you could find under `airflow/dags/etl_ftp_mongodb_customes_to_postgres.py`)

```Python3
def download_files_to_local():
    for url in FILE_DOWNLOADS_URLS:
        try:
            response = requests.get(url)
            # Save the downloaded file locally
            with open(f"local_{url.split('/')[-1]}", "wb") as f:
                f.write(response.content)
        except Exception as e:
            print(f"An error occurred: {e}")


def upload_files_to_minio():
    s3_hook = S3Hook(aws_conn_id=S3_CONN_ID)
    for filename in FILENAMES_HTTP:
        s3_hook.load_file(
            filename=filename,
            key=S3_TARGET_KEY,
            bucket_name=S3_CUSTOMERS_BUCKET,
            replace=True,
        )


def el_pdfs_from_http_to_s3():
    download_files_to_local()
    upload_files_to_minio()
```

This code is encapsulated in a task or node, the latter being is part of a pipeline. Each pipeline in Airflow is a "DAG", or Directed Acyclic Graph, a common terminology used in graph theory and data engineering. Each node in that graph is a processing step, and nodes are connected in a "one-way" fashion (hence the directed), and can't go back to the same node, otherwise infinite loop (hence the acyclic). Those connected nodes inherently represent dependencies between those nodes. For instance, it doesn't make sense for me to read data from the data lake to process and store in the data warehouse, if I haven't extracted the data from the source into the data lake in the first place. In a similar fashion, I don't need to wait for me to finish reading the data MongoDB so that I can start reading from the FTP server, hence those two tasks could be ran in parallel. 

Behind the scenes, multiple worker nodes will share those tasks by using a **broker** pattern (ref. ), representing asynchronous messaging. Mouthful, I know, but in short, imagine a queue of tasks, each waiting to be taken by our limited resources of workers. That lovely term, *scale*, that you probably hear every other day? Well, that's it. In this context, the Airflow scheduler sends the tasks to the message queue (e.g. Redis, RabbitMQ), which is later processed by workers (e.g. Celery). This processing could include, like the code above, accessing external resources, performing some calculations, and even triggering other processes.

[Credits: https://medium.com/sicara/using-airflow-with-celery-workers-54cb5212d405]


You will see such a setup commonly when working with distributed systems, the go-to approach when dealing with massive amounts of data or processing required.

Note, those concepts don't relate only to one technology but you'll see reused and reused as it's more part of a paradigm or framework than an isolated feature.

This concept of DAG is also used with Big Data technologies like Spark, and I encourage you to read more about it with this resource if you're interested.

Interesting Notes: 

* With such a data architecture, we are approaching what we call a **Data Fabric** paradigm.


1. Data Storage: Datalake where the raw data will reside (whether unstructured, semi-structured, or structured) to ensure its fidelity as well as data warehouse for more efficient querying.
2. Data Integration: Using an orchestrator, we'll develop ELT pipelines to ingest data from fragmented sources into one place e.g. Data Lake, as well as leverage dimensional data modeling / OLAP cubes (facts & dimensions) in our data warehouse to ensure the data is well structured and consistent
3. Business Intelligence: Develop interactive reports and dashboards for analyzing the data more visually. 
4. Data Governance: We establish processes for capturing metadata to provide more context for **discovery**, **lineage**, and **quality monitoring**, and compliance (access control, data encryption & anonimization, audit trails...)

## Data Storage

### Data Lake
### Data Warehouse

## Data Integration

### Airflow

I have gotten a question in a presentation: how can we ensrue as business users we can **trust** the data?

This si where ensuring we have proper data observability incl. lineage and cataloguing, come to ensure we know exactly what the data has been going through (tough life) from its ingestion. We ensure the data we have is first stored raw and avoid making a transformation to it. This way, if something doesn't make sense in the dashboard, we can trace back the  unexpected behaviour
and better pinpoint / root cause analyse. If the raw data makes sense, then you know it's got something to do at the source. 

You see, many of the life of a data engineer isn't to just build, but much of it is to maintain. When the data changes hands so much, and something goes wrong, it's easier to point fingers. This is why **data contracts** have come in place as well as proper
practices of versioning and quality monitoring to be able to detect those issues as soon as they happen as well as better find its root cause.