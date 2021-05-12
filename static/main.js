//Show classification buttons once video has loaded
$("#video-feed").on('load', function(){
    $('.loader').addClass('d-none');
    $('.btn-group').removeClass('d-none');
    $('.btn-group').addClass('d-block');
});

//Send classification data point to be saved
$(function(){
    $('button.classification').click(function(){
        value = $(this).attr('value');
        $.ajax({
            url: '/classify',
            data: value,
            type: 'POST',
            success: function(response){
                $(".alert-container").prepend("<div class='alert alert-success alert-dismissible fade show' role='alert'><strong>Success! </strong>" + response + "<button type='button' class='close' data-dismiss='alert' aria-label='Close'><span aria-hidden='true'>&times;</span></button></div>");
                $(".alert").delay(3000).slideUp(200, function() {
                    $(this).removeClass('show');
                });
            },
            error: function(error){
                console.log(error);
            }
        });
    });
});