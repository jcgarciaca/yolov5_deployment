import os
from lxml import etree
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import pandas as pd

def create_xml(info, df):
    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = 'Images'
    ET.SubElement(annotation, 'filename').text = info['img_name']
    ET.SubElement(annotation, 'path').text = info['img_name']
    source = ET.SubElement(annotation, 'source')
    ET.SubElement(source, 'database').text = 'Unknown'
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(info['img_size'][0])
    ET.SubElement(size, 'height').text = str(info['img_size'][1])
    ET.SubElement(size, 'depth').text = '3'
    ET.SubElement(annotation, 'segmented').text = '0'
    
    for _, row in df.iterrows():
        xmin = round(row['xmin'])
        ymin = round(row['ymin'])
        xmax = round(row['xmax'])
        ymax = round(row['ymax'])
        object_name = row['name']
        object_annotation = ET.SubElement(annotation, 'object')
        ET.SubElement(object_annotation, 'name').text = object_name
        ET.SubElement(object_annotation, 'pose').text = 'Unspecified'
        ET.SubElement(object_annotation, 'truncated').text = '0'
        ET.SubElement(object_annotation, 'difficult').text = '0'
        bndbox = ET.SubElement(object_annotation, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(xmin)
        ET.SubElement(bndbox, 'ymin').text = str(ymin)
        ET.SubElement(bndbox, 'xmax').text = str(xmax)
        ET.SubElement(bndbox, 'ymax').text = str(ymax)

    tree = ET.ElementTree(annotation)
    tree_root = tree.getroot()
    xmlstr = ET.tostring(tree_root, encoding='utf8', method='xml')
    dom = minidom.parseString(xmlstr)
    xml_object = dom.toprettyxml()
    
    with open(info['xml_path'], 'w') as writter:
        writter.write(xml_object)