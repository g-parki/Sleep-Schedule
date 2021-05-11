
$(function(){
    $('button.classification').click(function(){
        value = $(this).attr('value');
        $.ajax({
            url: '/classify',
            data: value,
            type: 'POST',
            success: function(response){
                $(".alert-container").prepend("<div class='alert alert-warning alert-dismissible fade show' role='alert'><strong>Success!</strong> Photo classified.<button type='button' class='close' data-dismiss='alert' aria-label='Close'><span aria-hidden='true'>&times;</span></button></div>");
                $(".alert").delay(2000).slideUp(200, function() {
                    $(this).removeClass('show');
                });
            },
            error: function(error){
                console.log(error);
            }
        });
    });
});