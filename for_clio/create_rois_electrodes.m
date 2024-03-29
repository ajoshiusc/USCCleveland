
function create_rois_electrodes(name,channel_file,input_mri_name,Radius,output_name)
ch_and_mris=load(channel_file);

%Read the specific Channel file for subject contained in 'channel_file'
formatSpec3 = '%s%s';
C1 = 'ch_';
chName=sprintf(formatSpec3,C1,name);
ch=ch_and_mris.(chName); %% using a struct and a dynamic string to access ch_and_mris

%Load sMRI data (prior surgery)
v=load_nii(input_mri_name);
vout=v;

%Read the specific mris file for subject contained in 'channel_file'
formatSpec4 = '%s%s';
D1 = 'mri_';
mriName=sprintf(formatSpec4,D1,name);
mriElect=ch_and_mris.(mriName);

vout.hdr=mriElect.Header;
vout.hdr.dime=vout.hdr.dim;
vout.hdr.dime.roi_scale=1;%v.hdr.dime.roi_scale
vout.hdr.hk=mriElect.Header.key;%v.hdr.hk;
vout.img= int16(zeros(size(mriElect.Cube)));
vout.hdr.hist.magic='n+1';

vout.hdr.hk.dim_info=6;
vout.hdr.dime.intent_p1=mriElect.Header.nifti.intent_p1;
vout.hdr.dime.intent_p2=mriElect.Header.nifti.intent_p2;
vout.hdr.dime.intent_p3=mriElect.Header.nifti.intent_p3;
vout.hdr.dime.intent_code=mriElect.Header.nifti.intent_code;
vout.hdr.dime=mriElect.Header.nifti;
vout.hdr.dime.datatype=vout.hdr.dim.datatype;
vout.hdr.dime.dim=vout.hdr.dim.dim;
vout.hdr.dime.pixdim=vout.hdr.dim.pixdim;
vout.hdr.dime.cal_max=vout.hdr.dim.cal_max;
vout.hdr.dime.cal_min=vout.hdr.dim.cal_min;
vout.hdr.dime.glmax=vout.hdr.dim.glmax;
vout.hdr.dime.glmin=vout.hdr.dim.glmin;
vout.hdr.hist=v.hdr.hist;



se = strel('sphere',Radius);
cc=zeros(200);
for i = 1:length(ch.Channel)

    loc=ch.Channel(i).Loc;
    
    electrodeloc = loc;
    
    if isempty(electrodeloc)
        continue;
    end
    
    electrodeloc = cs_convert(mriElect, 'SCS', 'mri', electrodeloc)*1000;
    res = mriElect.Voxsize;
    xx1 = electrodeloc(1)/res(1)+1;
    center = 121;
    xx1 = (center-xx1)+center;
    yy1 = electrodeloc(2)/res(2)+1;
    zz1 = electrodeloc(3)/res(3)+1;
    ind1 = sub2ind(size(mriElect.Cube),round(xx1),round(yy1),round(zz1));
    disp(i)
    cc(i)=i;
    vout.img(ind1)=i+100;
end
%%
vout.img=imdilate(vout.img,se);
save_untouch_nii(vout,output_name);

end





%%
function nii = load_nii(filename, img_idx, dim5_idx, dim6_idx, dim7_idx,old_RGB, tolerance, preferredForm)
%  Load NIFTI or ANALYZE dataset. Support both *.nii and *.hdr/*.img
%  file extension. If file extension is not provided, *.hdr/*.img will
%  be used as default.
%
%  A subset of NIFTI transform is included. For non-orthogonal rotation,
%  shearing etc., please use 'reslice_nii.m' to reslice the NIFTI file.
%  It will not cause negative effect, as long as you remember not to do
%  slice time correction after reslicing the NIFTI file. Output variable
%  nii will be in RAS orientation, i.e. X axis from Left to Right,
%  Y axis from Posterior to Anterior, and Z axis from Inferior to
%  Superior.
%
%  Usage: nii = load_nii(filename, [img_idx], [dim5_idx], [dim6_idx], ...
%			[dim7_idx], [old_RGB], [tolerance], [preferredForm])
%
%  filename  - 	NIFTI or ANALYZE file name.
%
%  img_idx (optional)  -  a numerical array of 4th dimension indices,
%	which is the indices of image scan volume. The number of images
%	scan volumes can be obtained from get_nii_frame.m, or simply
%	hdr.dime.dim(5). Only the specified volumes will be loaded.
%	All available image volumes will be loaded, if it is default or
%	empty.
%
%  dim5_idx (optional)  -  a numerical array of 5th dimension indices.
%	Only the specified range will be loaded. All available range
%	will be loaded, if it is default or empty.
%
%  dim6_idx (optional)  -  a numerical array of 6th dimension indices.
%	Only the specified range will be loaded. All available range
%	will be loaded, if it is default or empty.
%
%  dim7_idx (optional)  -  a numerical array of 7th dimension indices.
%	Only the specified range will be loaded. All available range
%	will be loaded, if it is default or empty.
%
%  old_RGB (optional)  -  a scale number to tell difference of new RGB24
%	from old RGB24. New RGB24 uses RGB triple sequentially for each
%	voxel, like [R1 G1 B1 R2 G2 B2 ...]. Analyze 6.0 from AnalyzeDirect
%	uses old RGB24, in a way like [R1 R2 ... G1 G2 ... B1 B2 ...] for
%	each slices. If the image that you view is garbled, try to set
%	old_RGB variable to 1 and try again, because it could be in
%	old RGB24. It will be set to 0, if it is default or empty.
%
%  tolerance (optional) - distortion allowed in the loaded image for any
%	non-orthogonal rotation or shearing of NIfTI affine matrix. If
%	you set 'tolerance' to 0, it means that you do not allow any
%	distortion. If you set 'tolerance' to 1, it means that you do
%	not care any distortion. The image will fail to be loaded if it
%	can not be tolerated. The tolerance will be set to 0.1 (10%), if
%	it is default or empty.
%
%  preferredForm (optional)  -  selects which transformation from voxels
%	to RAS coordinates; values are s,q,S,Q.  Lower case s,q indicate
%	"prefer sform or qform, but use others if preferred not present".
%	Upper case indicate the program is forced to use the specificied
%	tranform or fail loading.  'preferredForm' will be 's', if it is
%	default or empty.	- Jeff Gunter
%
%  Returned values:
%
%  nii structure:
%
%	hdr -		struct with NIFTI header fields.
%
%	filetype -	Analyze format .hdr/.img (0);
%			NIFTI .hdr/.img (1);
%			NIFTI .nii (2)
%
%	fileprefix - 	NIFTI filename without extension.
%
%	machine - 	machine string variable.
%
%	img - 		3D (or 4D) matrix of NIFTI data.
%
%	original -	the original header before any affine transform.
%
%  Part of this file is copied and modified from:
%  http://www.mathworks.com/matlabcentral/fileexchange/1878-mri-analyze-tools
%
%  NIFTI data format can be found on: http://nifti.nimh.nih.gov
%
%  - Jimmy Shen (jimmy@rotman-baycrest.on.ca)
%
if ~exist('filename','var')
    error('Usage: nii = load_nii(filename, [img_idx], [dim5_idx], [dim6_idx], [dim7_idx], [old_RGB], [tolerance], [preferredForm])');
end

if ~exist('img_idx','var') || isempty(img_idx)
    img_idx = [];
end

if ~exist('dim5_idx','var') || isempty(dim5_idx)
    dim5_idx = [];
end

if ~exist('dim6_idx','var') || isempty(dim6_idx)
    dim6_idx = [];
end

if ~exist('dim7_idx','var') || isempty(dim7_idx)
    dim7_idx = [];
end

if ~exist('old_RGB','var') || isempty(old_RGB)
    old_RGB = 0;
end

if ~exist('tolerance','var') || isempty(tolerance)
    tolerance = 0.1;			% 10 percent
end

if ~exist('preferredForm','var') || isempty(preferredForm)
    preferredForm= 's';		% Jeff
end

v = version;

%  Check file extension. If .gz, unpack it into temp folder
%
if length(filename) > 2 && strcmp(filename(end-2:end), '.gz')
    
    if ~strcmp(filename(end-6:end), '.img.gz') && ...
            ~strcmp(filename(end-6:end), '.hdr.gz') && ...
            ~strcmp(filename(end-6:end), '.nii.gz')
        
        error('Please check filename.');
    end
    
    if str2double(v(1:3)) < 7.1 || ~usejava('jvm')
        error('Please use MATLAB 7.1 (with java) and above, or run gunzip outside MATLAB.');
    elseif strcmp(filename(end-6:end), '.img.gz')
        filename1 = filename;
        filename2 = filename;
        filename2(end-6:end) = '';
        filename2 = [filename2, '.hdr.gz'];
        
        tmpDir = tempname;
        mkdir(tmpDir);
        gzFileName = filename;
        
        filename1 = gunzip(filename1, tmpDir);
        filename2 = gunzip(filename2, tmpDir);
        filename = char(filename1);	% convert from cell to string
    elseif strcmp(filename(end-6:end), '.hdr.gz')
        filename1 = filename;
        filename2 = filename;
        filename2(end-6:end) = '';
        filename2 = [filename2, '.img.gz'];
        
        tmpDir = tempname;
        mkdir(tmpDir);
        gzFileName = filename;
        
        filename1 = gunzip(filename1, tmpDir);
        filename2 = gunzip(filename2, tmpDir);
        filename = char(filename1);	% convert from cell to string
    elseif strcmp(filename(end-6:end), '.nii.gz')
        tmpDir = tempname;
        mkdir(tmpDir);
        gzFileName = filename;
        filename = gunzip(filename, tmpDir);
        filename = char(filename);	% convert from cell to string
    end
end

%  Read the dataset header
%
[nii.hdr,nii.filetype,nii.fileprefix,nii.machine] = load_nii_hdr(filename);

%  Read the header extension
%
%   nii.ext = load_nii_ext(filename);

%  Read the dataset body
%
[nii.img,nii.hdr] = load_nii_img(nii.hdr,nii.filetype,nii.fileprefix, ...
    nii.machine,img_idx,dim5_idx,dim6_idx,dim7_idx,old_RGB);

%  Perform some of sform/qform transform
%
%nii = xform_nii(nii, tolerance, preferredForm);
nii = load_untouch_nii(filename);

%  Clean up after gunzip
%
if exist('gzFileName', 'var')
    
    %  fix fileprefix so it doesn't point to temp location
    %
    nii.fileprefix = gzFileName(1:end-7);
    rmdir(tmpDir,'s');
end

return					% load_nii
end
%%
function [hdr, filetype, fileprefix, machine] = load_nii_hdr(fileprefix)
%  internal function

%  - Jimmy Shen (jimmy@rotman-baycrest.on.ca)
if ~exist('fileprefix','var'),
    error('Usage: [hdr, filetype, fileprefix, machine] = load_nii_hdr(filename)');
end
machine = 'ieee-le';
new_ext = 0;
if strfind(fileprefix,'.nii') && strcmp(fileprefix(end-3:end), '.nii')
    new_ext = 1;
    fileprefix(end-3:end)='';
    
elseif strfind(fileprefix,'.hdr') && strcmp(fileprefix(end-3:end), '.hdr')
    fileprefix(end-3:end)='';
    
elseif strfind(fileprefix,'.img') && strcmp(fileprefix(end-3:end), '.img')
    fileprefix(end-3:end)='';
end

if new_ext
    fn = sprintf('%s.nii',fileprefix);
    
    if ~exist(fn,'file')
        error('Cannot find file "%s.nii".', fileprefix);
    end
else
    fn = sprintf('%s.hdr',fileprefix);
    
    if ~exist(fn,'file')
        error('Cannot find file "%s.hdr".', fileprefix);
    end
end

fid = fopen(fn,'r',machine);

if fid < 0,
    error('Cannot open file %s.',fn);
else
    fseek(fid,0,'bof');
    
    if fread(fid,1,'int32') == 348
        hdr = read_header(fid);
        fclose(fid);
    else
        fclose(fid);
        
        %  first try reading the opposite endian to 'machine'
        switch machine,
            case 'ieee-le', machine = 'ieee-be';
            case 'ieee-be', machine = 'ieee-le';
        end
        
        fid = fopen(fn,'r',machine);
        
        if fid < 0,
            error('Cannot open file %s.',fn);
        else
            fseek(fid,0,'bof');
            
            if fread(fid,1,'int32') ~= 348
                %  Now throw an error
                error('File "%s" is corrupted.',fn);
            end
            
            hdr = read_header(fid);
            fclose(fid);
        end
    end
end

if strcmp(hdr.hist.magic, 'n+1')
    filetype = 2;
elseif strcmp(hdr.hist.magic, 'ni1')
    filetype = 1;
else
    filetype = 0;
end

return
end

function [ dsr ] = read_header(fid)
%  Original header structures
%  struct dsr
%       {
%       struct header_key hk;            /*   0 +  40       */
%       struct image_dimension dime;     /*  40 + 108       */
%       struct data_history hist;        /* 148 + 200       */
%       };                               /* total= 348 bytes*/

dsr.hk   = header_key(fid);
dsr.dime = image_dimension(fid);
dsr.hist = data_history(fid);

%  For Analyze data format
%
if ~strcmp(dsr.hist.magic, 'n+1') && ~strcmp(dsr.hist.magic, 'ni1')
    dsr.hist.qform_code = 0;
    dsr.hist.sform_code = 0;
end

return
end

function [ hk ] = header_key(fid)

fseek(fid,0,'bof');

%  Original header structures
%  struct header_key                     /* header key      */
%       {                                /* off + size      */
%       int sizeof_hdr                   /*  0 +  4         */
%       char data_type[10];              /*  4 + 10         */
%       char db_name[18];                /* 14 + 18         */
%       int extents;                     /* 32 +  4         */
%       short int session_error;         /* 36 +  2         */
%       char regular;                    /* 38 +  1         */
%       char dim_info;   % char hkey_un0;        /* 39 +  1 */
%       };                               /* total=40 bytes  */
%
% int sizeof_header   Should be 348.
% char regular        Must be 'r' to indicate that all images and
%                     volumes are the same size.

v6 = version;
if str2double(v6(1))<6
    directchar = '*char';
else
    directchar = 'uchar=>char';
end

hk.sizeof_hdr    = fread(fid, 1,'int32')';	% should be 348!
hk.data_type     = deblank(fread(fid,10,directchar)');
hk.db_name       = deblank(fread(fid,18,directchar)');
hk.extents       = fread(fid, 1,'int32')';
hk.session_error = fread(fid, 1,'int16')';
hk.regular       = fread(fid, 1,directchar)';
hk.dim_info      = fread(fid, 1,'uchar')';

return
end
%%
function [ dime ] = image_dimension(fid)

%  Original header structures
%  struct image_dimension
%       {                                /* off + size      */
%       short int dim[8];                /* 0 + 16          */
%       /*
%           dim[0]      Number of dimensions in database; usually 4.
%           dim[1]      Image X dimension;  number of *pixels* in an image row.
%           dim[2]      Image Y dimension;  number of *pixel rows* in slice.
%           dim[3]      Volume Z dimension; number of *slices* in a volume.
%           dim[4]      Time points; number of volumes in database
%       */
%       float intent_p1;   % char vox_units[4];   /* 16 + 4       */
%       float intent_p2;   % char cal_units[8];   /* 20 + 4       */
%       float intent_p3;   % char cal_units[8];   /* 24 + 4       */
%       short int intent_code;   % short int unused1;   /* 28 + 2 */
%       short int datatype;              /* 30 + 2          */
%       short int bitpix;                /* 32 + 2          */
%       short int slice_start;   % short int dim_un0;   /* 34 + 2 */
%       float pixdim[8];                 /* 36 + 32         */
%	/*
%		pixdim[] specifies the voxel dimensions:
%		pixdim[1] - voxel width, mm
%		pixdim[2] - voxel height, mm
%		pixdim[3] - slice thickness, mm
%		pixdim[4] - volume timing, in msec
%					..etc
%	*/
%       float vox_offset;                /* 68 + 4          */
%       float scl_slope;   % float roi_scale;     /* 72 + 4 */
%       float scl_inter;   % float funused1;      /* 76 + 4 */
%       short slice_end;   % float funused2;      /* 80 + 2 */
%       char slice_code;   % float funused2;      /* 82 + 1 */
%       char xyzt_units;   % float funused2;      /* 83 + 1 */
%       float cal_max;                   /* 84 + 4          */
%       float cal_min;                   /* 88 + 4          */
%       float slice_duration;   % int compressed; /* 92 + 4 */
%       float toffset;   % int verified;          /* 96 + 4 */
%       int glmax;                       /* 100 + 4         */
%       int glmin;                       /* 104 + 4         */
%       };                               /* total=108 bytes */

dime.dim        = fread(fid,8,'int16')';
dime.intent_p1  = fread(fid,1,'float32')';
dime.intent_p2  = fread(fid,1,'float32')';
dime.intent_p3  = fread(fid,1,'float32')';
dime.intent_code = fread(fid,1,'int16')';
dime.datatype   = fread(fid,1,'int16')';
dime.bitpix     = fread(fid,1,'int16')';
dime.slice_start = fread(fid,1,'int16')';
dime.pixdim     = fread(fid,8,'float32')';
dime.vox_offset = fread(fid,1,'float32')';
dime.scl_slope  = fread(fid,1,'float32')';
dime.scl_inter  = fread(fid,1,'float32')';
dime.slice_end  = fread(fid,1,'int16')';
dime.slice_code = fread(fid,1,'uchar')';
dime.xyzt_units = fread(fid,1,'uchar')';
dime.cal_max    = fread(fid,1,'float32')';
dime.cal_min    = fread(fid,1,'float32')';
dime.slice_duration = fread(fid,1,'float32')';
dime.toffset    = fread(fid,1,'float32')';
dime.glmax      = fread(fid,1,'int32')';
dime.glmin      = fread(fid,1,'int32')';

return
end

function [ hist ] = data_history(fid)

%  Original header structures
%  struct data_history
%       {                                /* off + size      */
%       char descrip[80];                /* 0 + 80          */
%       char aux_file[24];               /* 80 + 24         */
%       short int qform_code;            /* 104 + 2         */
%       short int sform_code;            /* 106 + 2         */
%       float quatern_b;                 /* 108 + 4         */
%       float quatern_c;                 /* 112 + 4         */
%       float quatern_d;                 /* 116 + 4         */
%       float qoffset_x;                 /* 120 + 4         */
%       float qoffset_y;                 /* 124 + 4         */
%       float qoffset_z;                 /* 128 + 4         */
%       float srow_x[4];                 /* 132 + 16        */
%       float srow_y[4];                 /* 148 + 16        */
%       float srow_z[4];                 /* 164 + 16        */
%       char intent_name[16];            /* 180 + 16        */
%       char magic[4];   % int smin;     /* 196 + 4         */
%       };                               /* total=200 bytes */

v6 = version;
if str2double(v6(1))<6
    directchar = '*char';
else
    directchar = 'uchar=>char';
end

hist.descrip     = deblank(fread(fid,80,directchar)');
hist.aux_file    = deblank(fread(fid,24,directchar)');
hist.qform_code  = fread(fid,1,'int16')';
hist.sform_code  = fread(fid,1,'int16')';
hist.quatern_b   = fread(fid,1,'float32')';
hist.quatern_c   = fread(fid,1,'float32')';
hist.quatern_d   = fread(fid,1,'float32')';
hist.qoffset_x   = fread(fid,1,'float32')';
hist.qoffset_y   = fread(fid,1,'float32')';
hist.qoffset_z   = fread(fid,1,'float32')';
hist.srow_x      = fread(fid,4,'float32')';
hist.srow_y      = fread(fid,4,'float32')';
hist.srow_z      = fread(fid,4,'float32')';
hist.intent_name = deblank(fread(fid,16,directchar)');
hist.magic       = deblank(fread(fid,4,directchar)');

fseek(fid,253,'bof');
hist.originator  = fread(fid, 5,'int16')';

return
end
%%
function [img,hdr] = load_nii_img(hdr,filetype,fileprefix,machine,img_idx,dim5_idx,dim6_idx,dim7_idx,old_RGB)
%  internal function

%  - Jimmy Shen (jimmy@rotman-baycrest.on.ca)
if ~exist('hdr','var') || ~exist('filetype','var') || ~exist('fileprefix','var') || ~exist('machine','var')
    error('Usage: [img,hdr] = load_nii_img(hdr,filetype,fileprefix,machine,[img_idx],[dim5_idx],[dim6_idx],[dim7_idx],[old_RGB]);');
end

if ~exist('img_idx','var') || isempty(img_idx) || hdr.dime.dim(5)<1
    img_idx = [];
end

if ~exist('dim5_idx','var') || isempty(dim5_idx) || hdr.dime.dim(6)<1
    dim5_idx = [];
end

if ~exist('dim6_idx','var') || isempty(dim6_idx) || hdr.dime.dim(7)<1
    dim6_idx = [];
end

if ~exist('dim7_idx','var') || isempty(dim7_idx) || hdr.dime.dim(8)<1
    dim7_idx = [];
end

if ~exist('old_RGB','var') || isempty(old_RGB)
    old_RGB = 0;
end

%  check img_idx
if ~isempty(img_idx) && ~isnumeric(img_idx)
    error('"img_idx" should be a numerical array.');
end

if length(unique(img_idx)) ~= length(img_idx)
    error('Duplicate image index in "img_idx"');
end

if ~isempty(img_idx) && (min(img_idx) < 1 || max(img_idx) > hdr.dime.dim(5))
    max_range = hdr.dime.dim(5);
    
    if max_range == 1
        error('"img_idx" should be 1.');
    else
        range = ['1 ' num2str(max_range)];
        error(['"img_idx" should be an integer within the range of [' range '].']);
    end
end

%  check dim5_idx
if ~isempty(dim5_idx) && ~isnumeric(dim5_idx)
    error('"dim5_idx" should be a numerical array.');
end

if length(unique(dim5_idx)) ~= length(dim5_idx)
    error('Duplicate index in "dim5_idx"');
end

if ~isempty(dim5_idx) && (min(dim5_idx) < 1 || max(dim5_idx) > hdr.dime.dim(6))
    max_range = hdr.dime.dim(6);
    
    if max_range == 1
        error('"dim5_idx" should be 1.');
    else
        range = ['1 ' num2str(max_range)];
        error(['"dim5_idx" should be an integer within the range of [' range '].']);
    end
end

%  check dim6_idx
if ~isempty(dim6_idx) && ~isnumeric(dim6_idx)
    error('"dim6_idx" should be a numerical array.');
end

if length(unique(dim6_idx)) ~= length(dim6_idx)
    error('Duplicate index in "dim6_idx"');
end

if ~isempty(dim6_idx) && (min(dim6_idx) < 1 || max(dim6_idx) > hdr.dime.dim(7))
    max_range = hdr.dime.dim(7);
    
    if max_range == 1
        error('"dim6_idx" should be 1.');
    else
        range = ['1 ' num2str(max_range)];
        error(['"dim6_idx" should be an integer within the range of [' range '].']);
    end
end

%  check dim7_idx
if ~isempty(dim7_idx) && ~isnumeric(dim7_idx)
    error('"dim7_idx" should be a numerical array.');
end

if length(unique(dim7_idx)) ~= length(dim7_idx)
    error('Duplicate index in "dim7_idx"');
end

if ~isempty(dim7_idx) && (min(dim7_idx) < 1 || max(dim7_idx) > hdr.dime.dim(8))
    max_range = hdr.dime.dim(8);
    
    if max_range == 1
        error('"dim7_idx" should be 1.');
    else
        range = ['1 ' num2str(max_range)];
        error(['"dim7_idx" should be an integer within the range of [' range '].']);
    end
end

[img,hdr] = read_image(hdr,filetype,fileprefix,machine,img_idx,dim5_idx,dim6_idx,dim7_idx,old_RGB);
return
end
%%
function [img,hdr] = read_image(hdr,filetype,fileprefix,machine,img_idx,dim5_idx,dim6_idx,dim7_idx,old_RGB)

switch filetype
    case {0, 1}
        fn = [fileprefix '.img'];
    case 2
        fn = [fileprefix '.nii'];
end

fid = fopen(fn,'r',machine);

if fid < 0,
    error('Cannot open file %s.',fn);
end

%  Set bitpix according to datatype
%
%  /*Acceptable values for datatype are*/
%
%     0 None                     (Unknown bit per voxel) % DT_NONE, DT_UNKNOWN
%     1 Binary                         (ubit1, bitpix=1) % DT_BINARY
%     2 Unsigned char         (uchar or uint8, bitpix=8) % DT_UINT8, NIFTI_TYPE_UINT8
%     4 Signed short                  (int16, bitpix=16) % DT_INT16, NIFTI_TYPE_INT16
%     8 Signed integer                (int32, bitpix=32) % DT_INT32, NIFTI_TYPE_INT32
%    16 Floating point    (single or float32, bitpix=32) % DT_FLOAT32, NIFTI_TYPE_FLOAT32
%    32 Complex, 2 float32      (Use float32, bitpix=64) % DT_COMPLEX64, NIFTI_TYPE_COMPLEX64
%    64 Double precision  (double or float64, bitpix=64) % DT_FLOAT64, NIFTI_TYPE_FLOAT64
%   128 uint8 RGB                 (Use uint8, bitpix=24) % DT_RGB24, NIFTI_TYPE_RGB24
%   256 Signed char            (schar or int8, bitpix=8) % DT_INT8, NIFTI_TYPE_INT8
%   511 Single RGB              (Use float32, bitpix=96) % DT_RGB96, NIFTI_TYPE_RGB96
%   512 Unsigned short               (uint16, bitpix=16) % DT_UNINT16, NIFTI_TYPE_UNINT16
%   768 Unsigned integer             (uint32, bitpix=32) % DT_UNINT32, NIFTI_TYPE_UNINT32
%  1024 Signed long long              (int64, bitpix=64) % DT_INT64, NIFTI_TYPE_INT64
%  1280 Unsigned long long           (uint64, bitpix=64) % DT_UINT64, NIFTI_TYPE_UINT64
%  1536 Long double, float128  (Unsupported, bitpix=128) % DT_FLOAT128, NIFTI_TYPE_FLOAT128
%  1792 Complex128, 2 float64  (Use float64, bitpix=128) % DT_COMPLEX128, NIFTI_TYPE_COMPLEX128
%  2048 Complex256, 2 float128 (Unsupported, bitpix=256) % DT_COMPLEX128, NIFTI_TYPE_COMPLEX128
%
switch hdr.dime.datatype
    case   1,
        hdr.dime.bitpix = 1;  precision = 'ubit1';
    case   2,
        hdr.dime.bitpix = 8;  precision = 'uint8';
    case   4,
        hdr.dime.bitpix = 16; precision = 'int16';
    case   8,
        hdr.dime.bitpix = 32; precision = 'int32';
    case  16,
        hdr.dime.bitpix = 32; precision = 'float32';
    case  32,
        hdr.dime.bitpix = 64; precision = 'float32';
    case  64,
        hdr.dime.bitpix = 64; precision = 'float64';
    case 128,
        hdr.dime.bitpix = 24; precision = 'uint8';
    case 256
        hdr.dime.bitpix = 8;  precision = 'int8';
    case 511
        hdr.dime.bitpix = 96; precision = 'float32';
    case 512
        hdr.dime.bitpix = 16; precision = 'uint16';
    case 768
        hdr.dime.bitpix = 32; precision = 'uint32';
    case 1024
        hdr.dime.bitpix = 64; precision = 'int64';
    case 1280
        hdr.dime.bitpix = 64; precision = 'uint64';
    case 1792,
        hdr.dime.bitpix = 128; precision = 'float64';
    otherwise
        error('This datatype is not supported');
end

hdr.dime.dim(hdr.dime.dim < 1) = 1;

%  move pointer to the start of image block
switch filetype
    case {0, 1}
        fseek(fid, 0, 'bof');
    case 2
        fseek(fid, hdr.dime.vox_offset, 'bof');
end

%  Load whole image block for old Analyze format or binary image;
%  otherwise, load images that are specified in img_idx, dim5_idx,
%  dim6_idx, and dim7_idx
%
%  For binary image, we have to read all because pos can not be
%  seeked in bit and can not be calculated the way below.

if hdr.dime.datatype == 1 || isequal(hdr.dime.dim(5:8),ones(1,4)) || ...
        (isempty(img_idx) && isempty(dim5_idx) && isempty(dim6_idx) && isempty(dim7_idx))
    
    %  For each frame, precision of value will be read
    %  in img_siz times, where img_siz is only the
    %  dimension size of an image, not the byte storage
    %  size of an image.
    img_siz = prod(hdr.dime.dim(2:8));
    
    %  For complex float32 or complex float64, voxel values
    %  include [real, imag]
    if hdr.dime.datatype == 32 || hdr.dime.datatype == 1792
        img_siz = img_siz * 2;
    end
    
    %MPH: For RGB24, voxel values include 3 separate color planes
    if hdr.dime.datatype == 128 || hdr.dime.datatype == 511
        img_siz = img_siz * 3;
    end
    
    img = fread(fid, img_siz, sprintf('*%s',precision));
    
    d1 = hdr.dime.dim(2);
    d2 = hdr.dime.dim(3);
    d3 = hdr.dime.dim(4);
    d4 = hdr.dime.dim(5);
    d5 = hdr.dime.dim(6);
    d6 = hdr.dime.dim(7);
    d7 = hdr.dime.dim(8);
    
    if isempty(img_idx)
        img_idx = 1:d4;
    end
    
    if isempty(dim5_idx)
        dim5_idx = 1:d5;
    end
    
    if isempty(dim6_idx)
        dim6_idx = 1:d6;
    end
    
    if isempty(dim7_idx)
        dim7_idx = 1:d7;
    end
else
    
    d1 = hdr.dime.dim(2);
    d2 = hdr.dime.dim(3);
    d3 = hdr.dime.dim(4);
    d4 = hdr.dime.dim(5);
    d5 = hdr.dime.dim(6);
    d6 = hdr.dime.dim(7);
    d7 = hdr.dime.dim(8);
    
    if isempty(img_idx)
        img_idx = 1:d4;
    end
    
    if isempty(dim5_idx)
        dim5_idx = 1:d5;
    end
    
    if isempty(dim6_idx)
        dim6_idx = 1:d6;
    end
    
    if isempty(dim7_idx)
        dim7_idx = 1:d7;
    end
    
    %  compute size of one image
    img_siz = prod(hdr.dime.dim(2:4));
    
    %  For complex float32 or complex float64, voxel values
    %  include [real, imag]
    if hdr.dime.datatype == 32 || hdr.dime.datatype == 1792
        img_siz = img_siz * 2;
    end
    
    %MPH: For RGB24, voxel values include 3 separate color planes
    if hdr.dime.datatype == 128 || hdr.dime.datatype == 511
        img_siz = img_siz * 3;
    end
    
    % preallocate img
    img = zeros(img_siz, length(img_idx)*length(dim5_idx)*length(dim6_idx)*length(dim7_idx) );
    currentIndex = 1;
    
    for i7=1:length(dim7_idx)
        for i6=1:length(dim6_idx)
            for i5=1:length(dim5_idx)
                for t=1:length(img_idx)
                    
                    %  Position is seeked in bytes. To convert dimension size
                    %  to byte storage size, hdr.dime.bitpix/8 will be
                    %  applied.
                    pos = sub2ind([d1 d2 d3 d4 d5 d6 d7], 1, 1, 1, ...
                        img_idx(t), dim5_idx(i5),dim6_idx(i6),dim7_idx(i7)) -1;
                    pos = pos * hdr.dime.bitpix/8;
                    
                    if filetype == 2
                        fseek(fid, pos + hdr.dime.vox_offset, 'bof');
                    else
                        fseek(fid, pos, 'bof');
                    end
                    
                    %  For each frame, fread will read precision of value
                    %  in img_siz times
                    img(:,currentIndex) = fread(fid, img_siz, sprintf('*%s',precision));
                    currentIndex = currentIndex +1;
                    
                end
            end
        end
    end
end

%  For complex float32 or complex float64, voxel values
%  include [real, imag]
if hdr.dime.datatype == 32 || hdr.dime.datatype == 1792
    img = reshape(img, [2, length(img)/2]);
    img = complex(img(1,:)', img(2,:)');
end

fclose(fid);

%  Update the global min and max values
hdr.dime.glmax = double(max(img(:)));
hdr.dime.glmin = double(min(img(:)));

%  old_RGB treat RGB slice by slice, now it is treated voxel by voxel
if old_RGB && hdr.dime.datatype == 128 && hdr.dime.bitpix == 24
    % remove squeeze
    img = (reshape(img, [hdr.dime.dim(2:3) 3 hdr.dime.dim(4) length(img_idx) length(dim5_idx) length(dim6_idx) length(dim7_idx)]));
    img = permute(img, [1 2 4 3 5 6 7 8]);
elseif hdr.dime.datatype == 128 && hdr.dime.bitpix == 24
    % remove squeeze
    img = (reshape(img, [3 hdr.dime.dim(2:4) length(img_idx) length(dim5_idx) length(dim6_idx) length(dim7_idx)]));
    img = permute(img, [2 3 4 1 5 6 7 8]);
elseif hdr.dime.datatype == 511 && hdr.dime.bitpix == 96
    img = double(img(:));
    img = single((img - min(img))/(max(img) - min(img)));
    % remove squeeze
    img = (reshape(img, [3 hdr.dime.dim(2:4) length(img_idx) length(dim5_idx) length(dim6_idx) length(dim7_idx)]));
    img = permute(img, [2 3 4 1 5 6 7 8]);
else
    % remove squeeze
    img = (reshape(img, [hdr.dime.dim(2:4) length(img_idx) length(dim5_idx) length(dim6_idx) length(dim7_idx)]));
end

if ~isempty(img_idx)
    hdr.dime.dim(5) = length(img_idx);
end

if ~isempty(dim5_idx)
    hdr.dime.dim(6) = length(dim5_idx);
end

if ~isempty(dim6_idx)
    hdr.dime.dim(7) = length(dim6_idx);
end

if ~isempty(dim7_idx)
    hdr.dime.dim(8) = length(dim7_idx);
end

return						% read_image
end
%%
function nii = load_untouch_nii(filename, img_idx, dim5_idx, dim6_idx, dim7_idx,old_RGB, slice_idx)
%  Load NIFTI or ANALYZE dataset, but not applying any appropriate affine
%  geometric transform or voxel intensity scaling.
%
%  Although according to NIFTI website, all those header information are
%  supposed to be applied to the loaded NIFTI image, there are some
%  situations that people do want to leave the original NIFTI header and
%  data untouched. They will probably just use MATLAB to do certain image
%  processing regardless of image orientation, and to save data back with
%  the same NIfTI header.
%
%  Since this program is only served for those situations, please use it
%  together with "save_untouch_nii.m", and do not use "save_nii.m" or
%  "view_nii.m" for the data that is loaded by "load_untouch_nii.m". For
%  normal situation, you should use "load_nii.m" instead.
%
%  Usage: nii = load_untouch_nii(filename, [img_idx], [dim5_idx], [dim6_idx], ...
%			[dim7_idx], [old_RGB], [slice_idx])
%
%  filename  - 	NIFTI or ANALYZE file name.
%
%  img_idx (optional)  -  a numerical array of image volume indices.
%	Only the specified volumes will be loaded. All available image
%	volumes will be loaded, if it is default or empty.
%
%	The number of images scans can be obtained from get_nii_frame.m,
%	or simply: hdr.dime.dim(5).
%
%  dim5_idx (optional)  -  a numerical array of 5th dimension indices.
%	Only the specified range will be loaded. All available range
%	will be loaded, if it is default or empty.
%
%  dim6_idx (optional)  -  a numerical array of 6th dimension indices.
%	Only the specified range will be loaded. All available range
%	will be loaded, if it is default or empty.
%
%  dim7_idx (optional)  -  a numerical array of 7th dimension indices.
%	Only the specified range will be loaded. All available range
%	will be loaded, if it is default or empty.
%
%  old_RGB (optional)  -  a scale number to tell difference of new RGB24
%	from old RGB24. New RGB24 uses RGB triple sequentially for each
%	voxel, like [R1 G1 B1 R2 G2 B2 ...]. Analyze 6.0 from AnalyzeDirect
%	uses old RGB24, in a way like [R1 R2 ... G1 G2 ... B1 B2 ...] for
%	each slices. If the image that you view is garbled, try to set
%	old_RGB variable to 1 and try again, because it could be in
%	old RGB24. It will be set to 0, if it is default or empty.
%
%  slice_idx (optional)  -  a numerical array of image slice indices.
%	Only the specified slices will be loaded. All available image
%	slices will be loaded, if it is default or empty.
%
%  Returned values:
%
%  nii structure:
%
%	hdr -		struct with NIFTI header fields.
%
%	filetype -	Analyze format .hdr/.img (0);
%			NIFTI .hdr/.img (1);
%			NIFTI .nii (2)
%
%	fileprefix - 	NIFTI filename without extension.
%
%	machine - 	machine string variable.
%
%	img - 		3D (or 4D) matrix of NIFTI data.
%
%  - Jimmy Shen (jimmy@rotman-baycrest.on.ca)
%

if ~exist('filename','var')
    error('Usage: nii = load_untouch_nii(filename, [img_idx], [dim5_idx], [dim6_idx], [dim7_idx], [old_RGB], [slice_idx])');
end

if ~exist('img_idx','var') || isempty(img_idx)
    img_idx = [];
end

if ~exist('dim5_idx','var') || isempty(dim5_idx)
    dim5_idx = [];
end

if ~exist('dim6_idx','var') || isempty(dim6_idx)
    dim6_idx = [];
end

if ~exist('dim7_idx','var') || isempty(dim7_idx)
    dim7_idx = [];
end

if ~exist('old_RGB','var') || isempty(old_RGB)
    old_RGB = 0;
end

if ~exist('slice_idx','var') || isempty(slice_idx)
    slice_idx = [];
end

v = version;

%  Check file extension. If .gz, unpack it into temp folder
if length(filename) > 2 && strcmp(filename(end-2:end), '.gz')
    
    if ~strcmp(filename(end-6:end), '.img.gz') && ...
            ~strcmp(filename(end-6:end), '.hdr.gz') && ...
            ~strcmp(filename(end-6:end), '.nii.gz')
        
        error('Please check filename.');
    end
    
    if str2double(v(1:3)) < 7.1 || ~usejava('jvm')
        error('Please use MATLAB 7.1 (with java) and above, or run gunzip outside MATLAB.');
    elseif strcmp(filename(end-6:end), '.img.gz')
        filename1 = filename;
        filename2 = filename;
        filename2(end-6:end) = '';
        filename2 = [filename2, '.hdr.gz'];
        
        tmpDir = tempname;
        mkdir(tmpDir);
        gzFileName = filename;
        
        filename1 = gunzip(filename1, tmpDir);
        filename2 = gunzip(filename2, tmpDir);
        filename = char(filename1);	% convert from cell to string
    elseif strcmp(filename(end-6:end), '.hdr.gz')
        filename1 = filename;
        filename2 = filename;
        filename2(end-6:end) = '';
        filename2 = [filename2, '.img.gz'];
        
        tmpDir = tempname;
        mkdir(tmpDir);
        gzFileName = filename;
        
        filename1 = gunzip(filename1, tmpDir);
        filename2 = gunzip(filename2, tmpDir);
        filename = char(filename1);	% convert from cell to string
    elseif strcmp(filename(end-6:end), '.nii.gz')
        tmpDir = tempname;
        mkdir(tmpDir);
        gzFileName = filename;
        filename = gunzip(filename, tmpDir);
        filename = char(filename);	% convert from cell to string
    end
end

%  Read the dataset header
[nii.hdr,nii.filetype,nii.fileprefix,nii.machine] = load_nii_hdr(filename);

if nii.filetype == 0
    nii.hdr = load_untouch_nii_hdr(nii.fileprefix,nii.machine);
    nii.ext = [];
else
    nii.hdr = load_untouch_nii_hdr(nii.fileprefix,nii.machine,nii.filetype);
    
    %  Read the header extension
    nii.ext = load_nii_ext(filename);
end

%  Read the dataset body
[nii.img,nii.hdr] = load_untouch_nii_img(nii.hdr,nii.filetype,nii.fileprefix, ...
    nii.machine,img_idx,dim5_idx,dim6_idx,dim7_idx,old_RGB,slice_idx);

%  Perform some of sform/qform transform
%
%   nii = xform_nii(nii, tolerance, preferredForm);

nii.untouch = 1;

%  Clean up after gunzip
if exist('gzFileName', 'var')
    
    %  fix fileprefix so it doesn't point to temp location
    nii.fileprefix = gzFileName(1:end-7);
    rmdir(tmpDir,'s');
end

return
end
%%
function hdr = load_untouch_nii_hdr(fileprefix, machine, filetype)
%  internal function

%  - Jimmy Shen (jimmy@rotman-baycrest.on.ca)

if filetype == 2
    fn = sprintf('%s.nii',fileprefix);
    
    if ~exist(fn,'file')
        error('Cannot find file "%s.nii".', fileprefix);
    end
else
    fn = sprintf('%s.hdr',fileprefix);
    
    if ~exist(fn,'file')
        error('Cannot find file "%s.hdr".', fileprefix);
    end
end

fid = fopen(fn,'r',machine);

if fid < 0,
    error('Cannot open file %s.',fn);
else
    fseek(fid,0,'bof');
    hdr = read_header_untouch(fid);
    fclose(fid);
end

return
end
%%
function [img,hdr] = read_image_untouch(hdr,filetype,fileprefix,machine,img_idx,dim5_idx,dim6_idx,dim7_idx,old_RGB,slice_idx)

switch filetype
    case {0, 1}
        fn = [fileprefix '.img'];
    case 2
        fn = [fileprefix '.nii'];
end

fid = fopen(fn,'r',machine);

if fid < 0,
    error('Cannot open file %s.',fn);
end

%  Set bitpix according to datatype
%
%  /*Acceptable values for datatype are*/
%
%     0 None                     (Unknown bit per voxel) % DT_NONE, DT_UNKNOWN
%     1 Binary                         (ubit1, bitpix=1) % DT_BINARY
%     2 Unsigned char         (uchar or uint8, bitpix=8) % DT_UINT8, NIFTI_TYPE_UINT8
%     4 Signed short                  (int16, bitpix=16) % DT_INT16, NIFTI_TYPE_INT16
%     8 Signed integer                (int32, bitpix=32) % DT_INT32, NIFTI_TYPE_INT32
%    16 Floating point    (single or float32, bitpix=32) % DT_FLOAT32, NIFTI_TYPE_FLOAT32
%    32 Complex, 2 float32      (Use float32, bitpix=64) % DT_COMPLEX64, NIFTI_TYPE_COMPLEX64
%    64 Double precision  (double or float64, bitpix=64) % DT_FLOAT64, NIFTI_TYPE_FLOAT64
%   128 uint8 RGB                 (Use uint8, bitpix=24) % DT_RGB24, NIFTI_TYPE_RGB24
%   256 Signed char            (schar or int8, bitpix=8) % DT_INT8, NIFTI_TYPE_INT8
%   511 Single RGB              (Use float32, bitpix=96) % DT_RGB96, NIFTI_TYPE_RGB96
%   512 Unsigned short               (uint16, bitpix=16) % DT_UNINT16, NIFTI_TYPE_UNINT16
%   768 Unsigned integer             (uint32, bitpix=32) % DT_UNINT32, NIFTI_TYPE_UNINT32
%  1024 Signed long long              (int64, bitpix=64) % DT_INT64, NIFTI_TYPE_INT64
%  1280 Unsigned long long           (uint64, bitpix=64) % DT_UINT64, NIFTI_TYPE_UINT64
%  1536 Long double, float128  (Unsupported, bitpix=128) % DT_FLOAT128, NIFTI_TYPE_FLOAT128
%  1792 Complex128, 2 float64  (Use float64, bitpix=128) % DT_COMPLEX128, NIFTI_TYPE_COMPLEX128
%  2048 Complex256, 2 float128 (Unsupported, bitpix=256) % DT_COMPLEX128, NIFTI_TYPE_COMPLEX128
%
switch hdr.dime.datatype
    case   1,
        hdr.dime.bitpix = 1;  precision = 'ubit1';
    case   2,
        hdr.dime.bitpix = 8;  precision = 'uint8';
    case   4,
        hdr.dime.bitpix = 16; precision = 'int16';
    case   8,
        hdr.dime.bitpix = 32; precision = 'int32';
    case  16,
        hdr.dime.bitpix = 32; precision = 'float32';
    case  32,
        hdr.dime.bitpix = 64; precision = 'float32';
    case  64,
        hdr.dime.bitpix = 64; precision = 'float64';
    case 128,
        hdr.dime.bitpix = 24; precision = 'uint8';
    case 256
        hdr.dime.bitpix = 8;  precision = 'int8';
    case 511
        hdr.dime.bitpix = 96; precision = 'float32';
    case 512
        hdr.dime.bitpix = 16; precision = 'uint16';
    case 768
        hdr.dime.bitpix = 32; precision = 'uint32';
    case 1024
        hdr.dime.bitpix = 64; precision = 'int64';
    case 1280
        hdr.dime.bitpix = 64; precision = 'uint64';
    case 1792,
        hdr.dime.bitpix = 128; precision = 'float64';
    otherwise
        error('This datatype is not supported');
end

tmp = hdr.dime.dim(2:end);
tmp(tmp < 1) = 1;
hdr.dime.dim(2:end) = tmp;

%  move pointer to the start of image block
switch filetype
    case {0, 1}
        fseek(fid, 0, 'bof');
    case 2
        fseek(fid, hdr.dime.vox_offset, 'bof');
end

%  Load whole image block for old Analyze format or binary image;
%  otherwise, load images that are specified in img_idx, dim5_idx,
%  dim6_idx, and dim7_idx
%
%  For binary image, we have to read all because pos can not be
%  seeked in bit and can not be calculated the way below.

if hdr.dime.datatype == 1 || isequal(hdr.dime.dim(4:8),ones(1,5)) || ...
        (isempty(img_idx) && isempty(dim5_idx) && isempty(dim6_idx) && isempty(dim7_idx) && isempty(slice_idx))
    
    %  For each frame, precision of value will be read
    %  in img_siz times, where img_siz is only the
    %  dimension size of an image, not the byte storage
    %  size of an image.
    img_siz = prod(hdr.dime.dim(2:8));
    
    %  For complex float32 or complex float64, voxel values
    %  include [real, imag]
    if hdr.dime.datatype == 32 || hdr.dime.datatype == 1792
        img_siz = img_siz * 2;
    end
    
    %MPH: For RGB24, voxel values include 3 separate color planes
    if hdr.dime.datatype == 128 || hdr.dime.datatype == 511
        img_siz = img_siz * 3;
    end
    
    img = fread(fid, img_siz, sprintf('*%s',precision));
    
    d1 = hdr.dime.dim(2);
    d2 = hdr.dime.dim(3);
    d3 = hdr.dime.dim(4);
    d4 = hdr.dime.dim(5);
    d5 = hdr.dime.dim(6);
    d6 = hdr.dime.dim(7);
    d7 = hdr.dime.dim(8);
    
    if isempty(slice_idx)
        slice_idx = 1:d3;
    end
    
    if isempty(img_idx)
        img_idx = 1:d4;
    end
    
    if isempty(dim5_idx)
        dim5_idx = 1:d5;
    end
    
    if isempty(dim6_idx)
        dim6_idx = 1:d6;
    end
    
    if isempty(dim7_idx)
        dim7_idx = 1:d7;
    end
else
    
    d1 = hdr.dime.dim(2);
    d2 = hdr.dime.dim(3);
    d3 = hdr.dime.dim(4);
    d4 = hdr.dime.dim(5);
    d5 = hdr.dime.dim(6);
    d6 = hdr.dime.dim(7);
    d7 = hdr.dime.dim(8);
    
    if isempty(slice_idx)
        slice_idx = 1:d3;
    end
    
    if isempty(img_idx)
        img_idx = 1:d4;
    end
    
    if isempty(dim5_idx)
        dim5_idx = 1:d5;
    end
    
    if isempty(dim6_idx)
        dim6_idx = 1:d6;
    end
    
    if isempty(dim7_idx)
        dim7_idx = 1:d7;
    end
    
    %ROMAN: begin
    roman = 1;
    if(roman)
        
        %  compute size of one slice
        img_siz = prod(hdr.dime.dim(2:3));
        
        %  For complex float32 or complex float64, voxel values
        %  include [real, imag]
        if hdr.dime.datatype == 32 || hdr.dime.datatype == 1792
            img_siz = img_siz * 2;
        end
        
        %MPH: For RGB24, voxel values include 3 separate color planes
        if hdr.dime.datatype == 128 || hdr.dime.datatype == 511
            img_siz = img_siz * 3;
        end
        
        % preallocate img
        img = zeros(img_siz, length(slice_idx)*length(img_idx)*length(dim5_idx)*length(dim6_idx)*length(dim7_idx) );
        currentIndex = 1;
    else
        img = [];
    end; %if(roman)
    % ROMAN: end
    
    for i7=1:length(dim7_idx)
        for i6=1:length(dim6_idx)
            for i5=1:length(dim5_idx)
                for t=1:length(img_idx)
                    for s=1:length(slice_idx)
                        
                        %  Position is seeked in bytes. To convert dimension size
                        %  to byte storage size, hdr.dime.bitpix/8 will be
                        %  applied.
                        %
                        pos = sub2ind([d1 d2 d3 d4 d5 d6 d7], 1, 1, slice_idx(s), ...
                            img_idx(t), dim5_idx(i5),dim6_idx(i6),dim7_idx(i7)) -1;
                        pos = pos * hdr.dime.bitpix/8;
                        
                        % ROMAN: begin
                        if(roman)
                            % do nothing
                        else
                            img_siz = prod(hdr.dime.dim(2:3));
                            
                            %  For complex float32 or complex float64, voxel values
                            %  include [real, imag]
                            %
                            if hdr.dime.datatype == 32 || hdr.dime.datatype == 1792
                                img_siz = img_siz * 2;
                            end
                            
                            %MPH: For RGB24, voxel values include 3 separate color planes
                            %
                            if hdr.dime.datatype == 128 || hdr.dime.datatype == 511
                                img_siz = img_siz * 3;
                            end
                        end; % if (roman)
                        % ROMAN: end
                        
                        if filetype == 2
                            fseek(fid, pos + hdr.dime.vox_offset, 'bof');
                        else
                            fseek(fid, pos, 'bof');
                        end
                        
                        %  For each frame, fread will read precision of value
                        %  in img_siz times
                        
                        % ROMAN: begin
                        if(roman)
                            img(:,currentIndex) = fread(fid, img_siz, sprintf('*%s',precision));
                            currentIndex = currentIndex +1;
                        else
                            img = [img fread(fid, img_siz, sprintf('*%s',precision))];
                        end; %if(roman)
                        % ROMAN: end
                        
                    end
                end
            end
        end
    end
end

%  For complex float32 or complex float64, voxel values
%  include [real, imag]
if hdr.dime.datatype == 32 || hdr.dime.datatype == 1792
    img = reshape(img, [2, length(img)/2]);
    img = complex(img(1,:)', img(2,:)');
end

fclose(fid);

%  Update the global min and max values
hdr.dime.glmax = double(max(img(:)));
hdr.dime.glmin = double(min(img(:)));

%  old_RGB treat RGB slice by slice, now it is treated voxel by voxel
if old_RGB && hdr.dime.datatype == 128 && hdr.dime.bitpix == 24
    % remove squeeze
    img = (reshape(img, [hdr.dime.dim(2:3) 3 length(slice_idx) length(img_idx) length(dim5_idx) length(dim6_idx) length(dim7_idx)]));
    img = permute(img, [1 2 4 3 5 6 7 8]);
elseif hdr.dime.datatype == 128 && hdr.dime.bitpix == 24
    % remove squeeze
    img = (reshape(img, [3 hdr.dime.dim(2:3) length(slice_idx) length(img_idx) length(dim5_idx) length(dim6_idx) length(dim7_idx)]));
    img = permute(img, [2 3 4 1 5 6 7 8]);
elseif hdr.dime.datatype == 511 && hdr.dime.bitpix == 96
    img = double(img(:));
    img = single((img - min(img))/(max(img) - min(img)));
    % remove squeeze
    img = (reshape(img, [3 hdr.dime.dim(2:3) length(slice_idx) length(img_idx) length(dim5_idx) length(dim6_idx) length(dim7_idx)]));
    img = permute(img, [2 3 4 1 5 6 7 8]);
else
    % remove squeeze
    img = (reshape(img, [hdr.dime.dim(2:3) length(slice_idx) length(img_idx) length(dim5_idx) length(dim6_idx) length(dim7_idx)]));
end

if ~isempty(slice_idx)
    hdr.dime.dim(4) = length(slice_idx);
end

if ~isempty(img_idx)
    hdr.dime.dim(5) = length(img_idx);
end

if ~isempty(dim5_idx)
    hdr.dime.dim(6) = length(dim5_idx);
end

if ~isempty(dim6_idx)
    hdr.dime.dim(7) = length(dim6_idx);
end

if ~isempty(dim7_idx)
    hdr.dime.dim(8) = length(dim7_idx);
end

return
end
%%
function [ dsr ] = read_header_untouch(fid)

%  Original header structures
%  struct dsr
%       {
%       struct header_key hk;            /*   0 +  40       */
%       struct image_dimension dime;     /*  40 + 108       */
%       struct data_history hist;        /* 148 + 200       */
%       };                               /* total= 348 bytes*/

dsr.hk   = header_key(fid);
dsr.dime = image_dimension(fid);
dsr.hist = data_history_untouch(fid);

%  For Analyze data format
if ~strcmp(dsr.hist.magic, 'n+1') && ~strcmp(dsr.hist.magic, 'ni1')
    dsr.hist.qform_code = 0;
    dsr.hist.sform_code = 0;
end

return					% read_header
end
%%
function [ hist ] = data_history_untouch(fid)

%  Original header structures
%  struct data_history
%       {                                /* off + size      */
%       char descrip[80];                /* 0 + 80          */
%       char aux_file[24];               /* 80 + 24         */
%       short int qform_code;            /* 104 + 2         */
%       short int sform_code;            /* 106 + 2         */
%       float quatern_b;                 /* 108 + 4         */
%       float quatern_c;                 /* 112 + 4         */
%       float quatern_d;                 /* 116 + 4         */
%       float qoffset_x;                 /* 120 + 4         */
%       float qoffset_y;                 /* 124 + 4         */
%       float qoffset_z;                 /* 128 + 4         */
%       float srow_x[4];                 /* 132 + 16        */
%       float srow_y[4];                 /* 148 + 16        */
%       float srow_z[4];                 /* 164 + 16        */
%       char intent_name[16];            /* 180 + 16        */
%       char magic[4];   % int smin;     /* 196 + 4         */
%       };                               /* total=200 bytes */

v6 = version;
if str2double(v6(1))<6
    directchar = '*char';
else
    directchar = 'uchar=>char';
end

hist.descrip     = deblank(fread(fid,80,directchar)');
hist.aux_file    = deblank(fread(fid,24,directchar)');
hist.qform_code  = fread(fid,1,'int16')';
hist.sform_code  = fread(fid,1,'int16')';
hist.quatern_b   = fread(fid,1,'float32')';
hist.quatern_c   = fread(fid,1,'float32')';
hist.quatern_d   = fread(fid,1,'float32')';
hist.qoffset_x   = fread(fid,1,'float32')';
hist.qoffset_y   = fread(fid,1,'float32')';
hist.qoffset_z   = fread(fid,1,'float32')';
hist.srow_x      = fread(fid,4,'float32')';
hist.srow_y      = fread(fid,4,'float32')';
hist.srow_z      = fread(fid,4,'float32')';
hist.intent_name = deblank(fread(fid,16,directchar)');
hist.magic       = deblank(fread(fid,4,directchar)');

return
end
%%
function ext = load_nii_ext(filename)
%  Load NIFTI header extension after its header is loaded using load_nii_hdr.
%
%  Usage: ext = load_nii_ext(filename)
%
%  filename - NIFTI file name.
%
%  Returned values:
%
%  ext - Structure of NIFTI header extension, which includes num_ext,
%       and all the extended header sections in the header extension.
%       Each extended header section will have its esize, ecode, and
%       edata, where edata can be plain text, xml, or any raw data
%       that was saved in the extended header section.
%
%  NIFTI data format can be found on: http://nifti.nimh.nih.gov
%
%  - Jimmy Shen (jimmy@rotman-baycrest.on.ca)
%
if ~exist('filename','var'),
    error('Usage: ext = load_nii_ext(filename)');
end


v = version;

%  Check file extension. If .gz, unpack it into temp folder
%
if length(filename) > 2 && strcmp(filename(end-2:end), '.gz')
    
    if ~strcmp(filename(end-6:end), '.img.gz') && ...
            ~strcmp(filename(end-6:end), '.hdr.gz') && ...
            ~strcmp(filename(end-6:end), '.nii.gz')
        
        error('Please check filename.');
    end
    
    if str2double(v(1:3)) < 7.1 || ~usejava('jvm')
        error('Please use MATLAB 7.1 (with java) and above, or run gunzip outside MATLAB.');
    elseif strcmp(filename(end-6:end), '.img.gz')
        filename1 = filename;
        filename2 = filename;
        filename2(end-6:end) = '';
        filename2 = [filename2, '.hdr.gz'];
        
        tmpDir = tempname;
        mkdir(tmpDir);
        gzFileName = filename;
        
        filename1 = gunzip(filename1, tmpDir);
        filename2 = gunzip(filename2, tmpDir);
        filename = char(filename1);	% convert from cell to string
    elseif strcmp(filename(end-6:end), '.hdr.gz')
        filename1 = filename;
        filename2 = filename;
        filename2(end-6:end) = '';
        filename2 = [filename2, '.img.gz'];
        
        tmpDir = tempname;
        mkdir(tmpDir);
        gzFileName = filename;
        
        filename1 = gunzip(filename1, tmpDir);
        filename2 = gunzip(filename2, tmpDir);
        filename = char(filename1);	% convert from cell to string
    elseif strcmp(filename(end-6:end), '.nii.gz')
        tmpDir = tempname;
        mkdir(tmpDir);
        gzFileName = filename;
        filename = gunzip(filename, tmpDir);
        filename = char(filename);	% convert from cell to string
    end
end

machine = 'ieee-le';
new_ext = 0;

if strfind(filename,'.nii') && strcmp(filename(end-3:end), '.nii')
    new_ext = 1;
    filename(end-3:end)='';
    
elseif strfind(filename,'.hdr') && strcmp(filename(end-3:end), '.hdr')
    filename(end-3:end)='';
    
elseif strfind(filename,'.img') && strcmp(filename(end-3:end), '.img')
    filename(end-3:end)='';
end

if new_ext
    fn = sprintf('%s.nii',filename);
    
    if ~exist(fn,'file')
        error('Cannot find file "%s.nii".', filename);
    end
else
    fn = sprintf('%s.hdr',filename);
    
    if ~exist(fn,'file')
        error('Cannot find file "%s.hdr".', filename);
    end
end

fid = fopen(fn,'r',machine);
vox_offset = 0;

if fid < 0,
    error('Cannot open file %s.',fn);
else
    fseek(fid,0,'bof');
    
    if fread(fid,1,'int32') == 348
        if new_ext
            fseek(fid,108,'bof');
            vox_offset = fread(fid,1,'float32');
        end
        
        ext = read_extension(fid, vox_offset);
        fclose(fid);
    else
        fclose(fid);
        
        %  first try reading the opposite endian to 'machine'
        %
        switch machine,
            case 'ieee-le', machine = 'ieee-be';
            case 'ieee-be', machine = 'ieee-le';
        end
        
        fid = fopen(fn,'r',machine);
        
        if fid < 0,
            error('Cannot open file %s.',fn);
        else
            fseek(fid,0,'bof');
            
            if fread(fid,1,'int32') ~= 348
                
                %  Now throw an error
                %
                error('File "%s" is corrupted.',fn);
            end
            
            if new_ext
                fseek(fid,108,'bof');
                vox_offset = fread(fid,1,'float32');
            end
            
            ext = read_extension(fid, vox_offset);
            fclose(fid);
        end
    end
end

%  Clean up after gunzip
if exist('gzFileName', 'var')
    rmdir(tmpDir,'s');
end

return
end
%%
function ext = read_extension(fid, vox_offset)

ext = [];

if vox_offset
    end_of_ext = vox_offset;
else
    fseek(fid, 0, 'eof');
    end_of_ext = ftell(fid);
end

if end_of_ext > 352
    fseek(fid, 348, 'bof');
    ext.extension = fread(fid,4)';
end

if isempty(ext) || ext.extension(1) == 0
    ext = [];
    return;
end

i = 1;

while(ftell(fid) < end_of_ext)
    ext.section(i).esize = fread(fid,1,'int32');
    ext.section(i).ecode = fread(fid,1,'int32');
    ext.section(i).edata = char(fread(fid,ext.section(i).esize-8)');
    i = i + 1;
end

ext.num_ext = length(ext.section);

return
end
%%
function [img,hdr] = load_untouch_nii_img(hdr,filetype,fileprefix,machine,img_idx,dim5_idx,dim6_idx,dim7_idx,old_RGB,slice_idx)
%  internal function

%  - Jimmy Shen (jimmy@rotman-baycrest.on.ca)
if ~exist('hdr','var') || ~exist('filetype','var') || ~exist('fileprefix','var') || ~exist('machine','var')
    error('Usage: [img,hdr] = load_nii_img(hdr,filetype,fileprefix,machine,[img_idx],[dim5_idx],[dim6_idx],[dim7_idx],[old_RGB],[slice_idx]);');
end

if ~exist('img_idx','var') || isempty(img_idx) || hdr.dime.dim(5)<1
    img_idx = [];
end

if ~exist('dim5_idx','var') || isempty(dim5_idx) || hdr.dime.dim(6)<1
    dim5_idx = [];
end

if ~exist('dim6_idx','var') || isempty(dim6_idx) || hdr.dime.dim(7)<1
    dim6_idx = [];
end

if ~exist('dim7_idx','var') || isempty(dim7_idx) || hdr.dime.dim(8)<1
    dim7_idx = [];
end

if ~exist('old_RGB','var') || isempty(old_RGB)
    old_RGB = 0;
end

if ~exist('slice_idx','var') || isempty(slice_idx) || hdr.dime.dim(4)<1
    slice_idx = [];
end

%  check img_idx
if ~isempty(img_idx) && ~isnumeric(img_idx)
    error('"img_idx" should be a numerical array.');
end

if length(unique(img_idx)) ~= length(img_idx)
    error('Duplicate image index in "img_idx"');
end

if ~isempty(img_idx) && (min(img_idx) < 1 || max(img_idx) > hdr.dime.dim(5))
    max_range = hdr.dime.dim(5);
    
    if max_range == 1
        error('"img_idx" should be 1.');
    else
        range = ['1 ' num2str(max_range)];
        error(['"img_idx" should be an integer within the range of [' range '].']);
    end
end

%  check dim5_idx
if ~isempty(dim5_idx) && ~isnumeric(dim5_idx)
    error('"dim5_idx" should be a numerical array.');
end

if length(unique(dim5_idx)) ~= length(dim5_idx)
    error('Duplicate index in "dim5_idx"');
end

if ~isempty(dim5_idx) && (min(dim5_idx) < 1 || max(dim5_idx) > hdr.dime.dim(6))
    max_range = hdr.dime.dim(6);
    
    if max_range == 1
        error('"dim5_idx" should be 1.');
    else
        range = ['1 ' num2str(max_range)];
        error(['"dim5_idx" should be an integer within the range of [' range '].']);
    end
end

%  check dim6_idx
if ~isempty(dim6_idx) && ~isnumeric(dim6_idx)
    error('"dim6_idx" should be a numerical array.');
end

if length(unique(dim6_idx)) ~= length(dim6_idx)
    error('Duplicate index in "dim6_idx"');
end

if ~isempty(dim6_idx) && (min(dim6_idx) < 1 || max(dim6_idx) > hdr.dime.dim(7))
    max_range = hdr.dime.dim(7);
    
    if max_range == 1
        error('"dim6_idx" should be 1.');
    else
        range = ['1 ' num2str(max_range)];
        error(['"dim6_idx" should be an integer within the range of [' range '].']);
    end
end

%  check dim7_idx
if ~isempty(dim7_idx) && ~isnumeric(dim7_idx)
    error('"dim7_idx" should be a numerical array.');
end

if length(unique(dim7_idx)) ~= length(dim7_idx)
    error('Duplicate index in "dim7_idx"');
end

if ~isempty(dim7_idx) && (min(dim7_idx) < 1 || max(dim7_idx) > hdr.dime.dim(8))
    max_range = hdr.dime.dim(8);
    
    if max_range == 1
        error('"dim7_idx" should be 1.');
    else
        range = ['1 ' num2str(max_range)];
        error(['"dim7_idx" should be an integer within the range of [' range '].']);
    end
end

%  check slice_idx
if ~isempty(slice_idx) && ~isnumeric(slice_idx)
    error('"slice_idx" should be a numerical array.');
end

if length(unique(slice_idx)) ~= length(slice_idx)
    error('Duplicate index in "slice_idx"');
end

if ~isempty(slice_idx) && (min(slice_idx) < 1 || max(slice_idx) > hdr.dime.dim(4))
    max_range = hdr.dime.dim(4);
    
    if max_range == 1
        error('"slice_idx" should be 1.');
    else
        range = ['1 ' num2str(max_range)];
        error(['"slice_idx" should be an integer within the range of [' range '].']);
    end
end

[img,hdr] = read_image_untouch(hdr,filetype,fileprefix,machine,img_idx,dim5_idx,dim6_idx,dim7_idx,old_RGB,slice_idx);

return
end

