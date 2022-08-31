# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""
import os.path

from apps.home import blueprint
from flask import render_template, request, jsonify
from flask_login import login_required
from jinja2 import TemplateNotFound

from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import imageio as iio


@blueprint.route('/index')
@login_required
def index():
    return render_template('home/landing-freelancer.html', segment='index')


@blueprint.route('/<template>')
@login_required
def route_template(template):
    try:

        if not template.endswith('.html'):
            template += '.html'

        # Detect the current page
        segment = get_segment(request)

        # Serve the file (if exists) from app/templates/home/FILE.html
        return render_template("home/" + template, segment=segment)

    except TemplateNotFound:
        return render_template('home/page-404.html'), 404

    except:
        return render_template('home/page-500.html'), 500


# Helper - Extract current page name from request
def get_segment(request):
    try:

        segment = request.path.split('/')[-1]

        if segment == '':
            segment = 'index'

        return segment

    except:
        return None


@blueprint.route('/get-predictions', methods=['POST'])
def get_predictions():
    global label
    classes = ['angular_leaf_spot', 'bean_rust', 'healthy']

    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    if request.method == 'POST':
        image = request.files['img']
        image_name = secure_filename(image.filename)
        image.save(os.path.join('uploads', image_name))

        model = tf.keras.models.load_model('models/')
        label = np.argmax(model.predict(decode_img(iio.imread(f'uploads/{image_name}'))), axis=1)

    return render_template('home/leaf-disease-prediction-results.html',
                           image_label=' '.join(classes[label[0]].split('_')).title())


def decode_img(image):
    img = tf.image.resize(image, [224, 224])
    img = tf.keras.preprocessing.image.img_to_array(img)
    return np.expand_dims(img, axis=0)
