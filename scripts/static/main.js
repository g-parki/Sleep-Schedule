$(document).ready(function(){
	$.ajaxSetup({ cache: false });
    
    //AJAX correct training data
    $('#tablecontainer').on('click','.training-correction', function(){
        $t = $(this);

        file_name = $(this).closest('div').attr('data-filename');
        value = $(this).attr('value');
        
        data_to_send = {
            file_name: file_name,
            value: value,
        };

        $.ajax({
            url: '/correcttrainingdata',
            contentType: "application/json",
            data: JSON.stringify(data_to_send),
            type: 'POST',
            success: function(response){
                console.log(response);
            },
            error: function(error){
                console.log(error);
            }
        }); 
    });

    //AJAX refresh training data table/pagination when pagination button is pressed
    $('#pagcontainer').on("click",'.page-item-ajax', function(){
        $t = $(this);
        value = $(this).attr('value');
        
        date = new Date;
    
        $.ajax({
            url: value,
            contentType: "application/json",
            data: JSON.stringify(date),
            type: 'POST',
            success: function(response){
                $('#pagcontainer').html(response['pagination_nav']);
                $('#tablecontainer').html(response['table']);
            },
            error: function(error){
                console.log(error);
            }
        });  
    });

    //Send classification data point to be saved from live stream
    $('button.live-classification').click(function(){
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
                
                date = new Date;
                $.ajax({
                    url: '/data?page=1&ajax=1',
                    contentType: "application/json",
                    data: JSON.stringify(date),
                    type: 'POST',
                    success: function(response){
                        $('#pagcontainer').html(response['pagination_nav']);
                        $('#tablecontainer').html(response['table']);
                    },
                    error: function(error){
                        console.log(error);
                    }
                });
            },
            error: function(error){
                console.log(error);
            }
        });
    });

    //Send classification data point to be saved
    $('button.correction').click(function(){
        $t = $(this);
        new_value = $(this).attr('value')

        data_to_send = {
            id: $(this).closest('tr').children('td.id-column').text(),
            value: new_value
        };
        $.ajax({
            url: '/correctdatapoint',
            contentType: "application/json",
            data: JSON.stringify(data_to_send),
            type: 'POST',
            success: function(response){
                if ( new_value == 'Empty'){
                    old_value = 'Baby';
                } else {
                    old_value = 'Empty';
                }
                $t.val(old_value);
                $t.text(old_value);
                $t.closest('div').siblings('button.dropdown-toggle').text(new_value);

                console.log(response);
            },
            error: function(error){
                console.log(error);
            }
        });
    });

    //Classify existing reading into training dataset
    $('button.reading-classification').click(function(){
        $t = $(this);
        value = $(this).attr('value');
        id = $(this).closest('tr').children('td.id-column').text();
        
        data_to_send = {
            id: id,
            value: value,
        };
        $.ajax({
            url: '/classifyreading',
            contentType: "application/json",
            data: JSON.stringify(data_to_send),
            type: 'POST',
            success: function(response){
                //Hide training buttons, show "Included in training data" div
                $t.closest('tr').find('div.add-to-training').addClass('d-none');
                $t.closest('tr').find('div.included-in').removeClass('d-none');
                console.log(response);
            },
            error: function(error){
                console.log(error);
            }
        });  
    });

    //Enable modal live stream on home page
    $('#playlivetext').click(function(){
        $('#myModal').modal();
    });
});

//Hide loader gif and show classification buttons when live stream loads
$("#video-feed").on('load', function(){
    $('.loader').addClass('d-none');
    $('.btn-group-live').removeClass('d-none');
    $('.btn-group-live').addClass('d-block');
    $('.bk').removeClass('d-none');
});