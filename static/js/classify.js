$(document).ready(function() {
  axios.get('/api/get_results')
  .then(response => {
      if (response.status === 200) {
          $('#results').text(response.data.log);
          console.log(response.data.log); // 在控制台打印返回结果
      }
  })
  .catch(error => {
      console.error(error);
  });
  $('#uploadForm').on('submit', function (event) {
      event.preventDefault();
      const formData = new FormData(this); // 收集表单数据
      // 显示黑色遮罩
      $('#overlay').show();
      axios.post('/upload', formData, {
          headers: {
              'Content-Type': 'multipart/form-data' // 设置正确的 Content-Type
          }
      })
      .then(response => {
          // 如果状态码为 200，显示成功信息
          if (response.status === 200) {
              // $('#result').css('color', 'green'); // 设置字体颜色为绿色
              // $('#results').text(`Success: ${response.data.message}`);
              // 在results文本框中追加新的结果
              $('#results').text(response.data.result + $('#results').text());
              console.log(response.data.result); // 在控制台打印返回结果
          }
      })
      .catch(error => {
           // 捕获错误并显示
          if (error.response) {
              // 处理后端返回的错误
              alert(`错误: ${error.response.status} - ${error.response.data.error}`);
          } else {
              // 处理网络错误或其他错误
              $('#result').text(`Error: ${error.message}`);
          }
      })
      .finally(() => {
          // 隐藏黑色遮罩
          $('#overlay').hide();
      });
  });
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
      $('#file-name').text("上传了" + files[0].name + (files.length > 1 ? "等文件" : "文件")); // 显示文件名
      this.val = files;
  }); 
    $('#select').click(() => {
        $('#file').value = '';
        $('#file-name').text('');
        $('#file').click();
  });
});