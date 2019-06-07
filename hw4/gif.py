from tqdm.auto import tqdm
import imageio

image_dir = './eval_images/acgan_500/'
writer = imageio.get_writer(image_dir + 'acgan_500.gif', fps=2)
for i in tqdm(range(0, 100, 5)):
    fname = image_dir + '%d.png' % (i)
    im = imageio.imread(fname)
    writer.append_data(im)
for i in tqdm(range(100, 500, 50)):
    fname = image_dir + '%d.png' % (i)
    im = imageio.imread(fname)
    writer.append_data(im)
writer.close()

image_dir = './eval_images/resnet_500/'
writer = imageio.get_writer(image_dir + 'resnet_500.gif', fps=2)
for i in tqdm(range(0, 100, 5)):
    fname = image_dir + '%d.png' % (i)
    im = imageio.imread(fname)
    writer.append_data(im)
for i in tqdm(range(100, 500, 50)):
    fname = image_dir + '%d.png' % (i)
    im = imageio.imread(fname)
    writer.append_data(im)
writer.close()

image_dir = './eval_images/resnet_hinge_500/'
writer = imageio.get_writer(image_dir + 'resnet_hinge_500.gif', fps=2)
for i in tqdm(range(0, 100, 5)):
    fname = image_dir + '%d.png' % (i)
    im = imageio.imread(fname)
    writer.append_data(im)
for i in tqdm(range(100, 500, 50)):
    fname = image_dir + '%d.png' % (i)
    im = imageio.imread(fname)
    writer.append_data(im)
writer.close()

image_dir = './eval_images/resnet_1000/'
writer = imageio.get_writer(image_dir + 'resnet_1000.gif', fps=2)
for i in tqdm(range(0, 100, 5)):
    fname = image_dir + '%d.png' % (i)
    im = imageio.imread(fname)
    writer.append_data(im)
for i in tqdm(range(100, 500, 50)):
    fname = image_dir + '%d.png' % (i)
    im = imageio.imread(fname)
    writer.append_data(im)
for i in tqdm(range(500, 1000, 100)):
    fname = image_dir + '%d.png' % (i)
    im = imageio.imread(fname)
    writer.append_data(im)
writer.close()