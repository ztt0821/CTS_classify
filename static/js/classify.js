$(document).ready(function () {
    let postEntries = [];
    let formData = new FormData();
    let filenames = [];
//   axios.get('/api/get_results')
//   .then(response => {
//       if (response.status === 200) {
//           $('#results').text(response.data.log);
//           console.log(response.data.log); // 在控制台打印返回结果
//       }
//   })
//   .catch(error => {
//       console.error(error);
//   });
  $('#uploadForm').on('submit', async function (event) {
      event.preventDefault();
    //   const formData = new FormData(this); // 收集表单数据
      // 显示黑色遮罩
    //   if (formData.get('file') === null) {
    //       alert('请选择至少一个文件');
    //       return;
    //   }
      $('#overlay').show();
      $('.result-content tbody').empty(); // 清空表格内容
      await processEntries(postEntries);
        $('#overlay').hide();
  });
  async function processEntries(postEntries) {
        for (let i = 0; i < postEntries.length; i++) {
            const entry = postEntries[i];
            filenames = []; // 清空文件名列表
            formData = new FormData(); // 清空formData
            await handleDirectory(entry);
            formData.set("parent_path", $('#parent_path').val());
            formData.set("filenames", JSON.stringify(filenames));
            // console.log(formData);
            try {
                const response = await axios.post('/upload', formData, {
                    headers: {
                        'Content-Type': 'multipart/form-data' // 设置正确的 Content-Type
                    }
                });
                if (response.status === 200) {
                    console.log(response.data.result); // 在控制台打印返回结果
                    res = response.data.result;
                    formData = new FormData(); // 清空formData
                    filenames = [];
                    const tableBody = $('.result-content tbody');

                    // 遍历文件夹和文件列表
                    $.each(res, function (_, item) {
                        const rowSpan = item.seq.length; // 该文件夹下文件数量
                        // 第一行显示文件夹名 + 第一个文件名
                        tableBody.append(`
                            <tr>
                                <td class="first_td" data-name="${item.name + '/' + item.paient_id}" data-res="${item.seq[0].class}" rowspan="${rowSpan}">${item.name}</td>
                                <td>${item.seq[0].name}</td>
                                <td>${item.seq[0].class == "normal" ? "低风险" : "高风险"}</td>
                                <td rowspan="${rowSpan}">${item.ok ? '否' : '是'}</td>
                            </tr> 
                        `);
                        // 剩余文件名占用新行
                        for (let i = 1; i < item.seq.length; i++) {
                            tableBody.append(`
                                <tr>
                                    <td>${item.seq[i].name}</td>
                                    <td>${item.seq[i].class == "normal" ? "低风险" : "高风险"}</td>
                                </tr>
                            `);
                        }
                    });
                }
            } catch (error) {
                // 捕获错误并显示
                if (error.response) {
                    // 处理后端返回的错误
                    alert(`错误: ${error.response.status} - ${error.response.data.error}`);
                } else {
                    // 处理网络错误或其他错误
                    $('#result').text(`Error: ${error.message}`);
                }
            }
        }
    }
  $("#logout").click(function() {
      axios.get('/logout')
      .then(response => {
          window.location.href = '/login';
      })
      .catch(error => {
          console.error(error);
      });
  });
  $('#file').change(e => {
      const files = e.target.files; // 获取选中的文件列表
    //   $('#file-name').text("上传了" + files[0].name + (files.length > 1 ? "等文件" : "文件")); // 显示文件名
      this.val = files;
  }); 
    $('#select').click(() => {
        $('#file').value = '';
        $('#file-name').text('');
        $('#file').click();
    });
    const dropZone = document.getElementById('dropZone');
    dropZone.addEventListener('dragover', (event) => {
        event.preventDefault();
        event.stopPropagation(); // 阻止事件冒泡
        dropZone.style.border = '2px dashed #00f'; 
    });
    dropZone.addEventListener('dragleave', () => {
        dropZone.style.border = '2px dashed #ccc';
    });
    dropZone.addEventListener('drop', (event) => {
        event.preventDefault();
        event.stopPropagation(); // 阻止事件冒泡
        // 发起遮罩
        $('#overlay').show();
        postEntries = [];
        $(".directories").empty();
        dropZone.style.border = '2px dashed #ccc';
        const items = event.dataTransfer.items;
        for (let i = 0; i < items.length; i++) {
            const item = items[i];
            if (item.kind === 'file') {
                const entry = item.webkitGetAsEntry();
                if (entry.isDirectory) {
                    postEntries.push(entry);
                    $(".directories").append(
                        `<div class="directory">
                        <img src="/static/images/folder.png" class="directory-image">
                        <span>${entry.name}</span>
                        </div>`
                    )
                }
            }
        }
        $('#overlay').hide();
    });
    function handleDirectory(dirEntry) {
        return new Promise((resolve, reject) => {
            const dirReader = dirEntry.createReader();
            dirReader.readEntries((entries) => {
                const promises = entries.map(entry => {
                    if (entry.isDirectory) {
                        return handleDirectory(entry); // 如果是目录，递归调用
                    } else {
                        // 判断文件后缀
                        endStr = entry.fullPath.slice(-4);
                        if (endStr == ".dcm") {
                            filenames.push(entry.fullPath); // 收集文件路径
                            console.log("找到文件", entry.fullPath);
                        }
                    }
                });
                // 等待所有递归调用的完成
                Promise.all(promises).then(resolve).catch(reject);
            }, (error) => {
                console.error("读取文件夹失败", error);
                reject(error); // 错误时拒绝 Promise
            });
        });
    }
    
    $(".download").click(() => {
        const _FormData = new FormData();
        const headers = ["病人名字", "病人编号", "检测结果", "时间"];
        _FormData.append("headers", JSON.stringify(headers));
        res = [];
        $(".first_td").each((_, item) => {
            var data_name = $(item).data("name");
            var name = data_name.split("/")[0];
            var paient_id = data_name.split("/")[1];
            res.push(JSON.stringify([name, paient_id, $(item).data("res"), new Date().toLocaleString()]));
        });
        _FormData.append("data", JSON.stringify(res));
        console.log("发送的数据", _FormData);
        axios.post('/download', _FormData, {
            responseType: 'blob'
        }).then(response => {
            const url = window.URL.createObjectURL(new Blob([response.data]));
            const link = document.createElement('a');
            link.href = url;
            link.setAttribute('download', 'result.xlsx');
            document.body.appendChild(link);
            link.click();
        }).catch(error => {
            console.error(error);
        });

    });
});