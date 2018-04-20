clear
addpath('script');
outdir = 'data/';

load_settings;


% AMAZON
load('data/amazon/amazon.mat');
splitsnum=[20];

for i=1:n_fold
tr20 = [];
te20 = [];

for k=1:n_class
    idx = find(label==k);
    idx = idx(randperm(numel(idx)));
    
    tr20 = [tr20;idx(1:20)];
    te  = idx(21:end);
    te20 = [te20;te];
    
    assert(numel(intersect(tr20,te20))==0);    
end
    
    
for m=1    
    eval(['tr_idx=tr',num2str(splitsnum(m)),';']);
    eval(['te_idx=te',num2str(splitsnum(m)),';']);   
    
    tr_files = cell(1,numel(tr_idx));
    te_files = cell(1,numel(te_idx));
    
    tr_label = label(tr_idx);
    assert(numel(unique(tr_label))==n_class);
    for z=1:numel(tr_idx)
        tr_files{z} = ['tr_out_n',num2str(tr_idx(z))];
    end    
    
    te_label = label(te_idx);
    assert(numel(unique(te_label))==n_class);
    for z=1:numel(te_idx)
        te_files{z} = ['te_out_n',num2str(te_idx(z))];
    end    

    splitfile = [outdir,'amazon/','amazon_split_F',num2str(i),'.mat'];
    save(splitfile,'tr_idx','te_idx','tr_label','te_label','tr_files','te_files');
    
    outfile = [outdir,'amazon/','amazon_cvsplit_F',num2str(i),'.mat'];
    [split.train_idx, split.test_idx] = create_cvsplit(tr_label, n_fold);
    
    save(outfile,'split');
    
end

end
clear fts labels;

% CALTECH
load('data/caltech/caltech.mat');
splitsnum=[20];

for i=1:n_fold
tr20 = [];
te20 = [];

for k=1:n_class
    idx = find(label==k);
    idx = idx(randperm(numel(idx)));
    
    tr20 = [tr20;idx(1:20)];
    te  = idx(21:end);
    te20 = [te20;te];
        
    assert(numel(intersect(tr20,te20))==0);
end

for m=1    
    eval(['tr_idx=tr',num2str(splitsnum(m)),';']);
    eval(['te_idx=te',num2str(splitsnum(m)),';']);
    
    tr_files = cell(1,numel(tr_idx));
    te_files = cell(1,numel(te_idx));
    
    tr_label = label(tr_idx);    
    assert(numel(unique(tr_label))==n_class);
    for z=1:numel(tr_idx)
        tr_files{z} = ['tr_out_n',num2str(tr_idx(z))];
    end    
    
    te_label = label(te_idx);
    assert(numel(unique(te_label))==n_class);
    for z=1:numel(te_idx)
        te_files{z} = ['te_out_n',num2str(te_idx(z))];
    end    
    
    splitfile = [outdir,'caltech/','caltech_split_F',num2str(i),'.mat'];
    save(splitfile,'tr_idx','te_idx','tr_label','te_label','tr_files','te_files');

    outfile = [outdir,'caltech/','caltech_cvsplit_F',num2str(i),'.mat'];
    [split.train_idx, split.test_idx] = create_cvsplit(tr_label, n_fold);
    save(outfile,'split');
end

end
clear fts labels;

% WEBCAM
load('data/webcam/webcam.mat');
splitsnum=[5];

for i=1:n_fold
tr5 = [];
te5 = [];

for k=1:n_class
    idx = find(label==k);
    idx = idx(randperm(numel(idx)));
    
    tr5 = [tr5;idx(1:5)];
    te  = idx(6:end);
    te5 = [te5;te];
        
    assert(numel(intersect(tr5,te5))==0);    
end

for m=1
    
    eval(['tr_idx=tr',num2str(splitsnum(m)),';']);
    eval(['te_idx=te',num2str(splitsnum(m)),';']);

    tr_files = cell(1,numel(tr_idx));
    te_files = cell(1,numel(te_idx));
    
    tr_label = label(tr_idx);    
    assert(numel(unique(tr_label))==n_class);    
    for z=1:numel(tr_idx)
        tr_files{z} = ['tr_out_n',num2str(tr_idx(z))];
    end
        
    te_label = label(te_idx);
    assert(numel(unique(te_label))==n_class);    
    for z=1:numel(te_idx)
        te_files{z} = ['te_out_n',num2str(te_idx(z))];
    end
    
    splitfile = [outdir,'webcam/','webcam_split_F',num2str(i),'.mat'];
    save(splitfile,'tr_idx','te_idx','tr_label','te_label','tr_files','te_files');

    outfile = [outdir,'webcam/','webcam_cvsplit_F',num2str(i),'.mat'];
    [split.train_idx, split.test_idx] = create_cvsplit(tr_label, n_fold);
    save(outfile,'split');
end

end
clear fts labels;


% DSLR
load('data/dslr/dslr.mat');
splitsnum=[5];

for i=1:n_fold
tr5 = [];
te5 = [];

for k=1:n_class
    idx = find(label==k);
    idx = idx(randperm(numel(idx)));
    
    tr5 = [tr5;idx(1:5)];
    te  = idx(6:end);
    te5 = [te5;te];
    
    assert(numel(intersect(tr5,te5))==0);    
end

for m=1    
    eval(['tr_idx=tr',num2str(splitsnum(m)),';']);
    eval(['te_idx=te',num2str(splitsnum(m)),';']);

    tr_files = cell(1,numel(tr_idx));
    te_files = cell(1,numel(te_idx));

    tr_label = label(tr_idx);    
    assert(numel(unique(tr_label))==n_class);    
    for z=1:numel(tr_idx)
        tr_files{z} = ['tr_out_n',num2str(tr_idx(z))];
    end
    
    te_label = label(te_idx);
    assert(numel(unique(te_label))==n_class);    
    for z=1:numel(te_idx)
        te_files{z} = ['te_out_n',num2str(te_idx(z))];
    end
    
    splitfile = [outdir,'dslr/','dslr_split_F',num2str(i),'.mat'];
    save(splitfile,'tr_idx','te_idx','tr_label','te_label','tr_files','te_files');
    
    outfile = [outdir,'dslr/','dslr_cvsplit_F',num2str(i),'.mat'];
    [split.train_idx, split.test_idx] = create_cvsplit(tr_label, n_fold);
    save(outfile,'split');
end


end
clear fts labels;

