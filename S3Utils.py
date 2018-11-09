import boto
import boto.s3.connection

access_key = u'57GSHTLHYV51PUJPA93X'
secret_key = u'tZ2q0Ed1VG8Ca2fTMMrt4i9kqgsW14J6f6tH40OA'
host = u"192.168.31.74"
port = 7480
bucket_name = "vsccapture"
conn = boto.connect_s3(aws_access_key_id=access_key,
                       aws_secret_access_key=secret_key,
                       host=host,
                       port=port,
                       is_secure=False,
                       calling_format=boto.s3.connection.OrdinaryCallingFormat(),
                       )


def getAllBuckets():
    for bucket in conn.get_all_buckets():
        print "{name}\t{created}".format(
            name=bucket.name,
            created=bucket.creation_date,
        )


def createBucket(bucketname):
    conn.create_bucket(bucketname)


def deleteBucket(bucketname):
    conn.delete_bucket(bucketname)


def downloadFile(filename):
    bucket = conn.get_bucket("vsccapture")
    key = bucket.get_key(filename)
    return key.get_contents_as_string()


def getBucket():
    bucket = conn.get_bucket(bucket_name)
    return bucket


def isBucketExisted(bucket_name):
    buckets = conn.get_all_buckets()
    for bucket in buckets:
        if bucket_name == bucket.name:
            return True
    return False


def upload(base64Str, file_name):
    bucket = conn.get_bucket(bucket_name)
    key = bucket.new_key(file_name)
    key.set_contents_from_string(image_byte)


def getUrl(file_name, period=3600, signed=True):
    bucket = getBucket()
    key = bucket.get_key(file_name)
    url = key.generate_url(period, query_auth=signed, force_http=False)
    return url


def deleteFile(file_name):
    bucket = getBucket()
    bucket.delete_key(file_name)


if __name__ == '__main__':
    print getUrl('test.jpg',)
