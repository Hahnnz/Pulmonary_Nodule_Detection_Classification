import os
import time
import cPickle
import datetime
import logging
import flask
import werkzeug
import optparse
import tornado.wsgi
import tornado.httpserver
import numpy as np
import pandas as pd
from PIL import Image
import cStringIO as StringIO
import urllib
import exifutil
import numpy as np
import cv2
import sys
caffe_root = '../caffe_o/'
sys.path.insert(0, caffe_root + 'python')
import caffe

global Sliced
Sliced=list()
global Slided
Slided=list()

REPO_DIRNAME = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../..')
UPLOAD_FOLDER = '/tmp/caffe_demos_uploads'
ALLOWED_IMAGE_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpe', 'jpeg', 'gif'])

model = '/home/hahnz/src/Prototype/TEST/deploy.prototxt'
pre = '/home/hahnz/src/Prototype/TEST/CNN-new3_iter_100000.caffemodel'
net = caffe.Net(model, pre, caffe.TEST)

# Obtain the flask app object
app = flask.Flask(__name__)


@app.route('/')
def index():
    return flask.render_template('index.html', has_result=False)


@app.route('/classify_url', methods=['GET'])
def classify_url():
    helloworld= cv2.imread("/home/hahnz/src/Prototype/TEST/check.jpg")
    imageurl = flask.request.args.get('imageurl', '')
    try:
        string_buffer = StringIO.StringIO(
            urllib.urlopen(imageurl).read())
        image = caffe.io.load_image(string_buffer)

    except Exception as err:
        # For any exception we encounter in reading the image, we will just
        # not continue.
        logging.info('URL Image open error: %s', err)
        return flask.render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open image from URL.')
        )

    logging.info('Image: %s', imageurl)
    result = app.clf.classify_image(image)
    helloworld=Predict(image,"Test","D")
    return flask.render_template(
        'index.html', has_result=True, result=result, imagesrc=helloworld)


@app.route('/classify_upload', methods=['POST'])
def classify_upload():
    try:
        # We will save the file to disk for possible data collection.
        imagefile = flask.request.files['imagefile']
        filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
            werkzeug.secure_filename(imagefile.filename)
        filename = os.path.join(UPLOAD_FOLDER, filename_)
        imagefile.save(filename)
        logging.info('Saving to %s.', filename)
	image = cv2.imread(filename)

    except Exception as err:
        logging.info('Uploaded image open error: %s', err)
        return flask.render_template(
            'index.html', has_result=True,
            result=(False, 'Cannot open uploaded image.')
        )
    result = app.clf.classify_image(image)
    helloworld=Predict(image,"Test","D")
    return flask.render_template(
        'index.html', has_result=True, result=result,
        #imagesrc=embed_image_html(image)
        imagesrc=embed_image_html(helloworld)
    )


def embed_image_html(image):
    """Creates an image embedded in HTML base64 format."""
    image_pil = Image.fromarray((image).astype('uint8'))
    string_buf = StringIO.StringIO()
    image_pil.save(string_buf, format='png')
    data = string_buf.getvalue().encode('base64').replace('\n', '')
    return 'data:image/png;base64,' + data

def SWL(img,time,Type): # Slice_Window = C / Slide_window = D
    if Type == "C":
        del Sliced[:]
        count=0 #(time*time)-1
        size=img.shape[0]
        x_axis , y_axis = img.shape[:2]
        crop_size=size/time
        for i in range(size/crop_size):
            for j in range(size/crop_size):
                crop_img=img[((crop_size)*i):((crop_size)*(i+1)), ((crop_size)*j):((crop_size)*(j+1))]
                Sliced.insert(count,crop_img)
                count+=1
        print "Slicing Done"
    elif Type == "D":
        del Slided[:]
        window = [0,63,0,63]
        count=0
        size=img.shape[0]
        x_axis , y_axis = img.shape[:2]
        
        while True :
            crop_img=img[window[0]:window[1]+1,window[2]:window[3]+1]
            Slided.insert(count,crop_img)
            count+=1
            window[0]+=512/time
            window[1]+=512/time
            if window[1] == 511:
                window[0]=0
                window[1]=63
                window[2]+=512/time
                window[3]+=512/time
            
            if window[3] == 511:
                break
        print "Sliding Done"       
        
    else:
        print "Please Check your layer type"
        print "Slice_Window = C / Slide_window = D"


def Masking(img,mask,countNum,Class,Type):
    hello = X_coor = Y_coor = 0
    while True:
        if Type == "C":
            if countNum<8:
                break
            countNum-=8
            hello+=64
            X_coor=hello
            Y_coor=64*countNum
        elif Type=="D":
            if countNum<28:
                break
            countNum-=28
            hello+=16
            X_coor=countNum*16
            Y_coor=hello
        
    if Class=="M":
        for i in range(64):
            for j in range(64):
                if Type=="C":
                    img[X_coor+i][Y_coor+j][0]=200
                elif img[X_coor+i][Y_coor+j][0]<255 and Type=="D":
                    #img[X_coor+i][Y_coor+j][0]+=15
                    mask[X_coor+i][Y_coor+j][0]+=1
    if Class=="B":
        for i in range(64):
            for j in range(64):
                if Type=="C":
                    img[X_coor+i][Y_coor+j][2]+=200
                elif img[X_coor+i][Y_coor+j][2]<255 and Type=="D":
                    #img[X_coor+i][Y_coor+j][2]+=15
                    mask[X_coor+i][Y_coor+j][2]+=1


def allowed_file(filename):
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1] in ALLOWED_IMAGE_EXTENSIONS
    )


def Predict(img,image_name,Type):
    Detected=cv2.imread("/home/hahnz/src/Prototype/TEST/check.jpg")
    result = img
    batch_size=128
    former=0
    Iter = 0
    time=32
    count =0
    
    SWL(img,time,Type)
    
    if Type=="C":
        Num=time*time/batch_size
    elif Type=="D":
        Num=len(Slided)/batch_size
    
    for Iter in range(Num) :
        for i in range(batch_size):
            if Type=="C":
                Input=Sliced[i+(Iter*batch_size)]
            elif Type == "D":
                Input=Slided[i+(Iter*batch_size)]
            im_input=Input[np.newaxis, np.newaxis :, :]
            img = im_input.transpose( (0,3,1,2) )
            net.blobs['data'].data[i][...] = img
            
        output=net.forward()
        
        for i in range(batch_size):
            a = net.blobs["fc2"].data[i][0]
            b = net.blobs["fc2"].data[i][1]
            c = net.blobs["fc2"].data[i][2]
            count+=1
            if b>a and b>c:
                Masking(result,Detected,i+(Iter*batch_size),"B",Type)
            elif c>a and c>b:
                Masking(result,Detected,i+(Iter*batch_size),"M",Type)
        if Type=="D":
            hello=Iter/10000
            if(former is not hello):
                former=hello
        count+=1
                
    for aa in range(512):
        for bb in range(512):
            if Detected[aa][bb][0] >10:
                result[aa][bb][0]+=200
            if Detected[aa][bb][2] >10:
                result[aa][bb][2]+=200
    return result

class ImagenetClassifier(object):
    default_args = {
        'model_def_file': (
            '{}/models/bvlc_reference_caffenet/deploy.prototxt'.format(REPO_DIRNAME)),
        'pretrained_model_file': (
            '{}/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'.format(REPO_DIRNAME)),
        'mean_file': (
            '{}/python/caffe/imagenet/ilsvrc_2012_mean.npy'.format(REPO_DIRNAME)),
        'class_labels_file': (
            '{}/data/ilsvrc12/synset_words.txt'.format(REPO_DIRNAME)),
        'bet_file': (
            '{}/data/ilsvrc12/imagenet.bet.pickle'.format(REPO_DIRNAME)),
    }
    for key, val in default_args.iteritems():
        if not os.path.exists(val):
            raise Exception(
                "File for {} is missing. Should be at: {}".format(key, val))
    default_args['image_dim'] = 512
    default_args['raw_scale'] = 512.

    def __init__(self, model_def_file, pretrained_model_file, mean_file,
                 raw_scale, class_labels_file, bet_file, image_dim, gpu_mode):
        logging.info('Loading net and associated files...')
        if gpu_mode:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()
        self.net = caffe.Classifier(
            model_def_file, pretrained_model_file,
            image_dims=(image_dim, image_dim), raw_scale=raw_scale,
            mean=np.load(mean_file).mean(1).mean(1), channel_swap=(2, 1, 0)
        )
        with open(class_labels_file) as f:
            labels_df = pd.DataFrame([
                {
                    'synset_id': l.strip().split(' ')[0],
                    'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
                }
                for l in f.readlines()
            ])
        self.labels = labels_df.sort_values('synset_id')['name'].values

        self.bet = cPickle.load(open(bet_file))
        # A bias to prefer children nodes in single-chain paths
        # I am setting the value to 0.1 as a quick, simple model.
        # We could use better psychological models here...
        self.bet['infogain'] -= np.array(self.bet['preferences']) * 0.1

    def classify_image(self, image):
        try:
            starttime = time.time()
            scores = self.net.predict([image], oversample=True).flatten()
            endtime = time.time()

            indices = (-scores).argsort()[:5]
            predictions = self.labels[indices]

            # In addition to the prediction text, we will also produce
            # the length for the progress bar visualization.
            meta = [
                (p, '%.5f' % scores[i])
                for i, p in zip(indices, predictions)
            ]
            logging.info('result: %s', str(meta))

            # Compute expected information gain
            expected_infogain = np.dot(
                self.bet['probmat'], scores[self.bet['idmapping']])
            expected_infogain *= self.bet['infogain']

            # sort the scores
            infogain_sort = expected_infogain.argsort()[::-1]
            bet_result = [(self.bet['words'][v], '%.5f' % expected_infogain[v])
                          for v in infogain_sort[:5]]
            logging.info('bet result: %s', str(bet_result))

            return (True, meta, bet_result, '%.3f' % (endtime - starttime))

        except Exception as err:
            logging.info('Classification error: %s', err)
            return (False, 'Something went wrong when classifying the '
                           'image. Maybe try another one?')


def start_tornado(app, port=5910):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()


def start_from_terminal(app):
    """
    Parse command line options and start the server.
    """
    parser = optparse.OptionParser()
    parser.add_option(
        '-d', '--debug',
        help="enable debug mode",
        action="store_true", default=False)
    parser.add_option(
        '-p', '--port',
        help="which port to serve content on",
        type='int', default=5910)
    parser.add_option(
        '-g', '--gpu',
        help="use gpu mode",
        action='store_true', default=False)

    opts, args = parser.parse_args()
    ImagenetClassifier.default_args.update({'gpu_mode': opts.gpu})

    # Initialize classifier + warm start by forward for allocation
    app.clf = ImagenetClassifier(**ImagenetClassifier.default_args)
    app.clf.net.forward()

    if opts.debug:
        app.run(debug=True, host='0.0.0.0', port=opts.port)
    else:
        start_tornado(app, opts.port)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    start_from_terminal(app)
