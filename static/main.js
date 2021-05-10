$(function(){
    $('button.classification').click(function(){
        value = $(this).attr('value');
        $.ajax({
            url: '/classify',
            data: value,
            type: 'POST',
            success: function(response){
                console.log(response);
                console.log(value);
            },
            error: function(error){
                console.log(error);
            }
        });
    });
});