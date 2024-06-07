import os
import xml.etree.ElementTree as ET 
import pandas as pd 

cvl_data_path = '../handwritten-text-recognition/cvl-database-1-1/'
root_dir = '../handwritten-text-recognition/'


def parseXML(file_path):
    with open(file_path, 'rb') as file:
            content = file.read()

    content = content.decode('utf-8', errors='replace')
    content = content.replace('encoding="ISO-8859-1"', 'encoding="utf-8"')

    # Parse the XML content
    root = ET.fromstring(content)
    namespace = {'ns': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-03-19'}
    lines = []

    # Find all AttrRegion elements with attrType=3 and fontType=2
    for region in root.findall('.//ns:AttrRegion[@attrType="3"][@fontType="2"]', namespace):
        # Find all child AttrRegion elements with attrType=2
        for sub_region in region.findall('.//ns:AttrRegion[@attrType="2"]', namespace):
            # Create a list to hold words for each line
            line = []

            # Find all child AttrRegion elements with attrType=1
            for word_region in sub_region.findall('.//ns:AttrRegion[@attrType="1"]', namespace):
                # Extract the text attribute value
                text = word_region.get('text')
                if text:
                    line.append(text)
            if line:
                lines.append(" ".join(line))

    return (lines, len(lines))


def make_split_lines(split):    
    df = {}
    for file_path in sorted(os.listdir(cvl_data_path+split+'xml')):
        lines, num_lines = parseXML(cvl_data_path+split+'xml/'+file_path)

        temp1 = file_path.split("_")[0]
        temp1 = [temp1]*num_lines
        temp2 = [str(i) for i in range(num_lines)]
        names = [x+'-'+y for x, y in zip(temp1, temp2)]

        df.update(zip(names, lines))

    df1 = pd.DataFrame(df.items(), columns=['file_name', 'text'])
    df1['file_name'] = df1['file_name'].apply(lambda x: x+'.tif')
    df1.to_csv(root_dir+"line_"+split+'.csv', index=False)
    

def make_split_page(split):
    df = {}
    for file_path in sorted(os.listdir(cvl_data_path+split+'xml')):
        lines, num_lines = parseXML(cvl_data_path+split+'xml/'+file_path)

        temp1 = file_path.split("_")[0]
        file_name = temp1 + '-cropped.tif'
        label = " ".join(lines)
        df.update({file_name:label})
        
    df1 = pd.DataFrame(df.items(), columns=['file_name', 'text'])
    df1.to_csv(root_dir+"page_"+split+'.csv', index=False)


make_split_lines('trainset')
make_split_lines('testset')

make_split_page('trainset')
make_split_page('testset')