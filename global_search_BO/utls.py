import yaml
import copy
import tempfile
import json
import os
import sys
sys.path.append("../vallhund")
sys.path.append("../fw3d_qpso")
from multiprocessing import Process
import itertools
import re
import numpy as np
import boto3
import time
from scipy.interpolate import Akima1DInterpolator as am
from scipy.interpolate import PchipInterpolator as pchip
from scipy.ndimage import shift, gaussian_filter
from vtrtool.aux_funcs import segymodelfile_to_ndarray
from importlib import reload
import segyio
import ipywidgets as widgets
import pandas as pd
import requests
from functools import lru_cache
from IPython.display import display, clear_output
from ipywidgets import interact, interactive
from vallhund.XWIJobs import *
from vallhund.utils.runfile.Runfile import Runfile
from vallhund.substrata import *
from vtrtool.aux_funcs import segymodelfile_to_ndarray, ndarray_to_segymodelfile

from particles.particle_base import Partices
from utilities.log_parser import parse_logs_for_shot_fit, parse_log_for_fs
from logger import QPSOLogger
from qpso import qpso


def list_jobs():
    """
    List active XWI Jobs in the table
    """
    active_jobs = sorted(list(JobRecord.scan(JobRecord.time_created.exists() & JobRecord.job_name.exists())),
                         key=lambda x: x.time_created, reverse=True)
    job_info = []
    for i, job in enumerate(active_jobs):
        # translate to local timezone
        job_time = job.time_created.replace(microsecond=0).astimezone(get_localzone())
        info = job.id + " | " + job.job_name.ljust(24, ' ') + '|' + str(job_time).rjust(30, ' ')
        job_info.append((info, job.id))
    return widgets.Select(options=job_info, description='Sector', disabled=False,
                          layout={'height':'150px', 'width':'50%'})

def list_active_jobs():
    """
    List active XWI Jobs in the table
    """
    active_jobs = sorted(get_active_jobs(), key=lambda x: x.time_created, reverse=True)
    job_info = []
    for i, active_job in enumerate(active_jobs):
        # translate to local timezone
        job = get_job_by_id(active_job.id)
        running = (job.scheduler_ip is not None) and (job.spot_fleet_id is not None)
        running_str = 'Running' if running else 'Queued'
        job_time = job.time_created.replace(microsecond=0).astimezone(get_localzone())
        info = job.id + " | " + job.job_name.ljust(24, ' ') + '|' + str(job_time).rjust(30, ' ') + '|' + \
        running_str.rjust(10, ' ')
        job_info.append((info, job.id))
    return widgets.Select(options=job_info, description='Sector', disabled=False,
                          layout={'height':'150px', 'width':'50%'})


def odered_dict_to_df(ordered_dict, row_size=3):
    entries = list(ordered_dict.items())
    df_dict = {}
    for i in range(0, len(entries), row_size):
        df_dict[i] = list(sum(entries[i:i+row_size], ()))
    return df_dict

def job_info(job_id):
    """
    Show basic info for a specific Job
    """
    job = get_job_by_id(job_id)

    # translate time to local timezone
    job_time = job.time_created.astimezone(get_localzone())
    info = ""

    info += f"Job {job_id}\n"
    info += "-------------------\n"
    info += f"job_name: {job.job_name}\n"
    info += f"user: {job.user}\n"
    info += f"active: {bool(job.active)}\n"
    info += f"time_created: {job_time}\n"
    info += f"unique_id: {job.unique_name}\n"
    info += f"job_basepath: {job.job_basepath}\n"
    info += f"output_path: {s3_path_from_basedir(job.output_path)}\n"
    info += f"scheduler_ec2_instance_id: {job.scheduler_ec2_instance_id}\n"
    info += f"scheduler_ip: {job.scheduler_ip}\n"
    info += f"spot_fleet_id: {job.spot_fleet_id}\n"

    if job.old_ec2_ids:
        info += f"old_ec2_ids: {job.old_ec2_ids}\n"
    if job.old_spot_fleet_ids:
        info += f"old_spot_fleet_ids: {job.old_spot_fleet_ids}\n"
    return info


def load_runfile(bucket, runfile):
    s3 = boto3.client("s3")
    res = s3.get_object(Bucket=bucket, Key=runfile)
    stream = res['Body']
    text = stream.read().decode("utf-8")
    runfile = Runfile.from_text(text)
    return runfile


def put_info(bucket, path, job_id):
    s3 = boto3.client("s3")
    res = s3.put_object(Bucket=bucket, Key=os.path.join(path, f'job_info{job_id}.txt'),
                        Body=job_info(job_id).encode('utf-8'))


def get_workspace():
    with open('/opt/ml/metadata/resource-metadata.json', 'r') as f:
        meta_data = json.load(f)
    self_arn = meta_data["ResourceArn"]
    client = boto3.client('sagemaker')
    response = client.list_tags(ResourceArn=self_arn)
    return [k["Value"] for k in response["Tags"] if k["Key"] == "VALLHUND_WORKSPACE"][0]


def get_sqs(region):
    workspace = get_workspace()
    client = boto3.client("ssm", region_name=region)
    regional_param_name = f"/SCubeConfigs/Vallhund/{workspace}/regional/configs"
    #regional_params = json.loads(ps.get_parameter(regional_param_name)["configs"])
    res = client.get_parameter(Name=regional_param_name, WithDecryption=True)
    return json.loads(res['Parameter']['Value'])['sqs_url']

def setup_projects(bucket, src_path, root_path, prefix, epsilon=None, num=0, title='A'):
    project_name = f"{title}{num}-epsilon_global"
    _copy_init_files(bucket, src_path, os.path.join(root_path, project_name), prefix, {"epsilon": epsilon})
    return project_name


def _copy_init_files(bucket, src_path, dest_path, prefix, settings):
    s3 = boto3.client("s3")
    setting_names = ['vp', 'delta', 'epsilon', None, 'wb']
    init_files = [prefix + name for name in ["-StartVp", "-TrueDelta", "-TrueEpsilon", "-TrueVp"]]
    init_files.append('water_bottom')
    names = [os.path.join(src_path, name) for name in init_files]
    names.append(os.path.join(src_path, 'water_bottom'))
    for src_prefix, name, key in zip(names, init_files, setting_names):
        matches = s3.list_objects_v2(Bucket=bucket, Prefix=src_prefix)
        contents = matches.get("Contents")
        if not contents:
            continue
        src_file = contents[0]["Key"]
        ext = os.path.splitext(src_file)[1]
        dest_file = os.path.join(dest_path, name + ext)
        if settings.get(key):
            with tempfile.NamedTemporaryFile() as tf:
                ndarray_to_segymodelfile(tf.name, settings[key])
                s3.put_object(Bucket=bucket, Key=dest_file, Body=tf.read())
        else:
            boto3.resource("s3").Object(bucket, dest_file).copy_from(CopySource={'Bucket': bucket, 'Key': src_file})


def create_job(template, root_path, epsilon=None, runfile_params=dict(), num=0, title='A'):
    if template.compressed:
        d = decompress_deserialize_string
    else:
        d = deserialize_string
    config = yaml.safe_load(d(template.job_config))
    src_path = config.get('model_path') or config.get('input_path')

    job_name = setup_projects(config["s3_bucket"], src_path, root_path, config["xwi_prefix"], epsilon, num, title)

    ef = tempfile.NamedTemporaryFile('w+')
    rf = tempfile.NamedTemporaryFile('w+')
    cf = tempfile.NamedTemporaryFile('w+')
    config['environment_variables'] = ef.name
    config['xwi_runfile'] = rf.name
    config['model_path'] = os.path.join(root_path, job_name)
    config['output_path'] = os.path.join(root_path, job_name)
    config['job_name'] = job_name
    with open(ef.name, "w") as f:
        f.write(d(template.environ_script))
    with open(rf.name, "w") as f:
        f.write(Runfile.from_text(d(template.orig_runfile_content)).generate_runfile(runfile_params))
    with open(cf.name, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    j_id = create_xwi_job(cf.name)
    return j_id

def get_segy(bucket, key, dims):
    s3 = boto3.client("s3")
    s3.download_file(Bucket=bucket, Key=key, Filename='/tmp/sgy_tmp.sgy')
    return segymodelfile_to_ndarray('/tmp/sgy_tmp.sgy', dims)


def get_horizon(vp, criteria):
    horizon = np.zeros((vp.shape[0], vp.shape[1]))
    for i in range(vp.shape[0]):
        for j in range(vp.shape[1]):
            horizon[i][j] = np.nonzero(vp[i, j] >= criteria)[0].min()
    return horizon

def form_model_base(water_layer, vp_ref, dims, pos, params, criteria, fill_value):
    model = (np.zeros(dims))
    max_depth = int(get_horizon(vp_ref, criteria).max())
    min_depth = int(water_layer.min())
    positions = np.array([min_depth, pos, (max_depth-min_depth)*0.5, max_depth+1])
    coeffis = np.array([0, params[0], params[1], 0])
    try:
        line = pchip(positions, coeffis)(np.arange(min_depth, max_depth))
    except ValueError as e:
        print(e, positions)
        raise(e)
    for i in range(dims[0]):
        for j in range(dims[1]):
            w = water_layer[i, j]
            shift(line, w-min_depth, model[i, j, min_depth:max_depth], cval=0, order=3)
    model = np.where(vp_ref < criteria, model, fill_value)
    return gaussian_filter(model, 3, mode='nearest', truncate=3)

def get_segy(bucket, key, dims):
    s3 = boto3.client("s3")
    s3.download_file(Bucket=bucket, Key=key, Filename='/tmp/sgy_tmp.sgy')
    return segymodelfile_to_ndarray('/tmp/sgy_tmp.sgy', dims)


def get_horizon(vp, criteria):
    horizon = np.zeros((vp.shape[0], vp.shape[1]))
    for i in range(vp.shape[0]):
        for j in range(vp.shape[1]):
            horizon[i][j] = np.nonzero(vp[i, j] >= criteria)[0].min()
    return horizon

def form_model_base(water_layer, vp_ref, dims, pos, params, criteria, fill_value):
    model = (np.zeros(dims))
    max_depth = int(get_horizon(vp_ref, criteria).max())
    min_depth = int(water_layer.min())
    positions = np.array([min_depth, pos, (max_depth-min_depth)*0.5, max_depth+1])
    coeffis = np.array([0, params[0], params[1], 0])
    try:
        line = pchip(positions, coeffis)(np.arange(min_depth, max_depth))
    except ValueError as e:
        print(e, positions)
        raise(e)
    for i in range(dims[0]):
        for j in range(dims[1]):
            w = water_layer[i, j]
            shift(line, w-min_depth, model[i, j, min_depth:max_depth], cval=0, order=3)
    model = np.where(vp_ref < criteria, model, fill_value)
    return gaussian_filter(model, 3, mode='nearest', truncate=3)

class ParamsParticle(Partices):
    def __init__(self, template, dims, params, dest_path, id_, title='A'):
        self.template = template
        self.dims = dims
        self.params = params
        self.dest_path = dest_path
        self.id = id_
        self.title = title
        # self.criteria = criteria
        self.criteria = None
        self.fit_args = dict()

    def fit(self):
        id_ = int(self.id.split("_")[-1])
        time.sleep(id_)
        testing_params = {p:v for (p, v) in zip(self.params, self.value)}
        self.pred_model = create_job(self.template, self.dest_path, runfile_params=testing_params,
                                     num=self.id, title=self.title)
        if self.pred_model is None:
            raise IOError("Cannot create job with new vel model at directory: {}".format(dest_path))
        start_xwi_scheduler(self.pred_model)
        launch_xwi_spot_workers(self.pred_model)

    def compute_loss(self):
        job = get_job_by_id(self.pred_model)
        scheduler_id = job.scheduler_ec2_instance_id
        while(job.active):
            time.sleep(60)
            job = get_job_by_id(self.pred_model)
        group_name = cwl_group_name("Dev-Ningzhi")
        stream_name = cwl_job_output_stream_name(self.pred_model, job.unique_name)
        logs = tail_cloudwatchlogs(group_name, stream_name, 60 * 24 * 365, print_out=False, region=job.region)
        logs = "\n".join([l['message'] for l in logs])
        with tempfile.NamedTemporaryFile('w+') as f:
            f.write(logs)
            fits = parse_logs_for_shot_fit([f.name])
            fnl = parse_log_for_fs([f.name])
        last_iter = max(fits.keys())
        traces_fit = np.mean(fits[last_iter])
        fnl1 = fnl[last_iter]
        if self.fit_args.get("trace_fit"):
            return 100.0-traces_fit, fnl1
        else:
            return fnl1, 100.0-traces_fit



